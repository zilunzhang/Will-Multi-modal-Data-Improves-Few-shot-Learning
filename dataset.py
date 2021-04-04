import os
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, ToTensor
from collections import OrderedDict
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.transforms import Categorical, Rotation
from torchmeta.transforms.splitters import Splitter
from torchmeta.transforms.utils import apply_wrapper
from utils import *
import pickle as pkl
from torchmeta.utils.data import BatchMetaDataLoader


def ClassSplitter(task=None, *args, **kwargs):
        return apply_wrapper(ClassSplitter_(*args, **kwargs), task)


class ClassSplitter_(Splitter):
    def __init__(self, shuffle=True, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None,
                 num_support_per_class=None, num_query_per_class=None,
                 random_state_seed=0):
        """
        Transforms a dataset into train/test splits for few-shot learning tasks,
        based on a fixed number of samples per class for each split. This is a
        dataset transformation to be applied as a `dataset_transform` in a
        `MetaDataset`.

        Parameters
        ----------
        shuffle : bool (default: `True`)
            Shuffle the data in the dataset before the split.

        num_samples_per_class : dict, optional
            Dictionary containing the names of the splits (as keys) and the
            corresponding number of samples per class in each split (as values).
            If not `None`, then the arguments `num_train_per_class`,
            `num_test_per_class`, `num_support_per_class` and
            `num_query_per_class` are ignored.

        num_train_per_class : int, optional
            Number of samples per class in the training split. This corresponds
            to the number of "shots" in "k-shot learning". If not `None`, this
            creates an item `train` for each task.

        num_test_per_class : int, optional
            Number of samples per class in the test split. If not `None`, this
            creates an item `test` for each task.

        num_support_per_class : int, optional
            Alias for `num_train_per_class`. If `num_train_per_class` is not
            `None`, then this argument is ignored. If not `None`, this creates
            an item `support` for each task.

        num_query_per_class : int, optional
            Alias for `num_test_per_class`. If `num_test_per_class` is not
            `None`, then this argument is ignored. If not `None`, this creates
            an item `query` for each task.

        random_state_seed : int, optional
            seed of the np.RandomState. Defaults to '0'.
        """
        self.shuffle = shuffle

        if num_samples_per_class is None:
            num_samples_per_class = OrderedDict()
            if num_train_per_class is not None:
                num_samples_per_class['train'] = num_train_per_class
            elif num_support_per_class is not None:
                num_samples_per_class['support'] = num_support_per_class
            if num_test_per_class is not None:
                num_samples_per_class['test'] = num_test_per_class
            elif num_query_per_class is not None:
                num_samples_per_class['query'] = num_query_per_class
        assert len(num_samples_per_class) > 0

        self._min_samples_per_class = sum(num_samples_per_class.values())
        super(ClassSplitter_, self).__init__(num_samples_per_class, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        for name, class_indices in all_class_indices.items():
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                dataset_indices = np.arange(num_samples)
                num_make_up = self._min_samples_per_class - num_samples
                data_make_up = np.random.choice(dataset_indices, num_make_up)
                dataset_indices = np.append(dataset_indices, data_make_up)
            else:
                dataset_indices = np.arange(num_samples)

            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(dataset_indices)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        for dataset in task.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class:
                dataset_indices = np.arange(num_samples)
                num_make_up = self._min_samples_per_class - num_samples
                data_make_up = np.random.choice(dataset_indices, num_make_up)
                dataset_indices = np.append(dataset_indices, data_make_up)

            else:
                dataset_indices = np.arange(num_samples)

            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend(split_indices + cum_size)
                ptr += num_split
            cum_size += num_samples

        return indices


class Custom(CombinationMetaDataset):
    """
    Overall FSL dataset
    """
    def __init__(self,
                 sampling_policy,
                 id_to_sentence,
                 sentence_to_id,
                 num_classes_per_task=None,
                 meta_train=False,
                 meta_val=False,
                 meta_test=False,
                 meta_split=None,
                 transform=None,
                 target_transform=None,
                 dataset_transform=None,
                 class_augmentations=None,
                 download=False
                 ):
        dataset = CustomClassDataset(sampling_policy,
                                     id_to_sentence,
                                     sentence_to_id,
                                     meta_train=meta_train,
                                     meta_val=meta_val,
                                     meta_test=meta_test,
                                     meta_split=meta_split,
                                     transform=transform,
                                     class_augmentations=class_augmentations,
                                     download=download
                                     )
        super(Custom, self).__init__(dataset,
                                     num_classes_per_task,
                                     target_transform=target_transform,
                                     dataset_transform=dataset_transform
                                     )


class CustomClassDataset(ClassDataset):
    """
    Get class from full dataset (dict: class id -> list of nori )
    """
    def __init__(self,
                 sampling_policy,
                 id_to_sentence,
                 sentence_to_id,
                 meta_train=False,
                 meta_val=False,
                 meta_test=False,
                 meta_split=None,
                 transform=None,
                 class_augmentations=None,
                 download=False
                 ):
        super(CustomClassDataset, self).__init__(meta_train=meta_train,
                                                 meta_val=meta_val,
                                                 meta_test=meta_test,
                                                 meta_split=meta_split,
                                                 class_augmentations=class_augmentations
                                                 )

        self.sampling_policy = sampling_policy
        self.transform = transform
        self.id_to_sentence = id_to_sentence
        self.sentence_to_id = sentence_to_id

        self.data, self._num_classes = self.split_by_policy()

    def split_by_policy(self):

        if self.meta_train:
            data_selected = self.sampling_policy["train"]
            class_selected_count = len(data_selected.keys())
        elif self.meta_val:
            data_selected = self.sampling_policy["val"]
            class_selected_count = len(data_selected.keys())
        elif self.meta_test:
            data_selected = self.sampling_policy["test"]
            class_selected_count = len(data_selected.keys())
        else:
            print("need to set one of 'meta_train', 'meta_valid', 'meta_test' to be True")

        self.labels = np.reshape(np.array(list(data_selected.keys())), (-1, ))

        return data_selected, class_selected_count

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CustomDataset(index,
                             data,
                             class_name,
                             self.id_to_sentence,
                             self.sentence_to_id,
                             transform=transform,
                             target_transform=target_transform
                             )

    @property
    def num_classes(self):
        return self._num_classes

    def _check_integrity(self):
        return True

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None


class CustomDataset(Dataset):
    """
    Get image from a class of data
    """
    def __init__(self,
                 index,
                 data,
                 class_name,
                 id_to_sentence,
                 sentence_to_id,
                 transform=None,
                 target_transform=None
                 ):
        super(CustomDataset, self).__init__(index,
                                            transform=transform,
                                            target_transform=target_transform
                                            )

        self.data = data
        self.id_to_sentence = id_to_sentence
        self.sentence_to_id = sentence_to_id
        self.class_name = class_name

    def __len__(self):
        return len(self.data["text"])

    def __getitem__(self, index):
        text_data = self.data["text"][index]
        image_data = self.data["img"][index]
        if image_data.shape != (84, 84, 3):
            w, h = image_data.shape
            place_holder = np.empty((w, h, 3), dtype=np.uint8)
            place_holder[:, :, 0] = image_data
            place_holder[:, :, 1] = image_data
            place_holder[:, :, 2] = image_data
            image_data = place_holder
        target = self.class_name

        if self.transform is not None:
            image_data = self.transform(Image.fromarray(np.uint8(image_data)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        encoded_list = []
        for sentence in text_data:
            encoded_result = np.int(self.sentence_to_id[sentence])
            encoded_list.append(encoded_result)
        text_data = np.array(encoded_list)

        return image_data, text_data, target


def helper_with_default(klass,
                        id_to_sentence,
                        sentence_to_id,
                        sampling_policy,
                        shots,
                        ways,
                        shuffle=True,
                        test_shots=None,
                        seed=None,
                        defaults={},
                        **kwargs):
    if 'num_classes_per_task' in kwargs:

        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', ToTensor())
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = defaults.get('target_transform',
                                                  Categorical(ways))
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)
    if test_shots is None:
        test_shots = shots
    dataset = klass(sampling_policy,
                    id_to_sentence,
                    sentence_to_id,
                    num_classes_per_task=ways,
                    **kwargs
                    )
    dataset = ClassSplitter(dataset,
                            shuffle=shuffle,
                            num_train_per_class=shots,
                            num_test_per_class=test_shots
                            )
    dataset.seed(seed)
    return dataset


def custom_dataset(
                  sampling_policy,
                  id_to_sentence,
                  sentence_to_id,
                  shots,
                  ways,
                  shuffle=True,
                  test_shots=None,
                  seed=None,
                  **kwargs
                  ):
    """Helper function to create a meta-dataset for the Mini-Imagenet dataset.

    Parameters
    ----------

    sampling_policy: sampling policy dictionary

    folder : string
        Root directory where the dataset folder `miniimagenet` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `MiniImagenet` class.

    See also
    --------
    `datasets.MiniImagenet` : Meta-dataset for the Mini-Imagenet dataset.
    """
    defaults = {
        'transform': Compose([Resize(84), ToTensor()])
    }

    return helper_with_default(
        Custom,
        id_to_sentence,
        sentence_to_id,
        sampling_policy,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def main():
    with open(os.path.join("pkl", "data.pkl"), "rb") as f:
        data = pkl.load(f)
        print()
        f.close()
    with open(os.path.join("pkl", "id_sentence_encoder.pkl"), "rb") as f:
        id_to_sentence = pkl.load(f)
        print()
        f.close()
    with open(os.path.join("pkl", "sentence_id_encoder.pkl"), "rb") as f:
        sentence_to_id = pkl.load(f)
        print()
        f.close()
    dataset = custom_dataset(
            data,
            id_to_sentence,
            sentence_to_id,
            ways=5,
            shots=1,
            test_shots=15,
            meta_train=True,
            download=False,
            seed=0,
            transform=imagenet_transform(stage='train')
    )

    dataloader = BatchMetaDataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        num_workers=0
    )

    for (batch_idx, batch) in enumerate(dataloader):
        support_img, support_text, support_labels = batch["train"]
        query_img, query_text, query_labels = batch["test"]
        print()


if __name__ == '__main__':
    main()
