import numpy as np
import pytorch_lightning as pl
import argparse
from trainer import FSLTrainer
from utils import mannul_seed_everything, imagenet_transform
from torch.utils.data import DataLoader
import yaml
from dataset import *
from pytorch_lightning.callbacks import ModelCheckpoint
import random


def get_fused_feature(config, selected_imgs, selected_texts, id_sentence_mapping, sentence_id_mapping):

    mannul_seed_everything(config['seed'])

    if config['num_gpu'] > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = "cuda"
    else:
        device = "cpu"
    print("ckpt file: {}".format(config["ckpt_file"]))
    fsl_trainer = FSLTrainer.load_from_checkpoint(config['ckpt_file'])
    print("fsl trainer loaded")
    fsl_trainer = fsl_trainer.to("cuda")
    fsl_trainer.set_config(config)
    fusion_method = "mean"

    image_feats = torch.zeros(len(selected_imgs), 128).to(device)
    text_feats = torch.zeros(len(selected_imgs), 128).to(device)

    i = 0
    while i < len(selected_imgs):
        selected_img = torch.from_numpy(np.array(selected_imgs[i], dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        selected_text = selected_texts[i]
        encoded_list = []
        for sentence in selected_text:
            encoded_result = np.int(sentence_id_mapping[sentence])
            encoded_list.append(encoded_result)
        text_data = torch.from_numpy(np.array(encoded_list)).unsqueeze(0).to(device)
        img_feat = fsl_trainer.image_backbone(selected_img, fusion_method)
        text_feat = fsl_trainer.text_backbone(text_data, id_sentence_mapping, fusion_method)
        image_feats[i] = img_feat
        text_feats[i] = text_feat
        print()
        i += 1
    fused_feature = (image_feats + text_feats) / 2
    return fused_feature.detach().cpu().numpy()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="config.yaml",
                        help='path of task file')
    parser.add_argument('--ckpt_file', type=str, default="multimodal_best_weights/_ckpt_epoch_489.ckpt",
                                    help='path of ckpt file')

    parser.add_argument('--test_size', type=str, default=600)
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    parser.add_argument('--num_cpu', type=int, default=16,
                        help='number of cpu per trail')
    parser.add_argument('--dataset_root', type=str,
                        default='../pkl_dataset', help='dataset root')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of episode per batch')

    args = parser.parse_args()
    print(os.getcwd())
    config = dict()
    with open(args.task_file) as file:
        task_yaml = yaml.load(file, Loader=yaml.FullLoader)

    config['dataset_root'] = args.dataset_root
    config["num_way"] = task_yaml["FSL_INFO"]["N"]
    config["num_shot"] = task_yaml["FSL_INFO"]["K"]
    config["num_query"] = task_yaml["FSL_INFO"]["Q"]
    config["seed"] = 0
    config["num_gpu"] = args.num_gpu
    config["num_cpu"] = args.num_cpu
    config["ckpt_file"] = args.ckpt_file
    config["batch_size"] = args.batch_size
    config["test_size"] = args.test_size

    with open(os.path.join(config["dataset_root"], "data.pkl"), "rb") as f:
        data = pkl.load(f)
        f.close()
    with open(os.path.join(config["dataset_root"], "id_sentence_encoder.pkl"), "rb") as f:
        id_to_sentence = pkl.load(f)
        print()
        f.close()
    with open(os.path.join(config["dataset_root"], "sentence_id_encoder.pkl"), "rb") as f:
        sentence_to_id = pkl.load(f)
        print()
        f.close()

    config['id_to_sentence'] = id_to_sentence
    config['sentence_to_id'] = sentence_to_id
    config["lr"] = 0.01
    all_classes = np.array(list(data["test"].keys()))
    N = 5
    K = 10
    selected_imgs = []
    selected_texts = []
    selected_classes = np.random.permutation(all_classes)[:N]
    labels = []

    for selected_class in selected_classes:
        labels.append(selected_class)
        class_img_content = data["test"][selected_class]["img"]
        class_text_content = data["test"][selected_class]["text"]
        random_idxs = np.random.permutation(np.arange(len(class_img_content)))[:K]
        for idx in random_idxs:
            selected_class_image_content = class_img_content[idx]
            selected_class_text_content = class_text_content[idx]
            selected_imgs.append(selected_class_image_content)
            selected_texts.append(selected_class_text_content)

    with open("visual_imgs_data.pkl", "wb") as f_img:
        pkl.dump(selected_imgs, f_img)
        f_img.close()
    with open("visual_texts_data.pkl", "wb") as f_text:
        pkl.dump(selected_texts, f_text)
        f_text.close()
    with open("labels.pkl", "wb") as f_label:
        pkl.dump(labels, f_label)
        f_label.close()

    with open("visual_imgs_data.pkl", "rb") as f_img:
        selected_imgs = pkl.load(f_img)
        f_img.close()

    with open("visual_texts_data.pkl", "rb") as f_text:
        selected_texts = pkl.load(f_text)
        f_text.close()

    fused_feature = get_fused_feature(config, selected_imgs, selected_texts, id_to_sentence, sentence_to_id)
    print(fused_feature.shape)
    np.save("fused_feature", fused_feature)
    print(fused_feature.shape)


def plot_feat():
    fused_feat = np.load("fused_feature.npy")
    img_feat = np.load("image_feature.npy")
    labels = pkl.load(open("labels.pkl", "rb"))
    real_labels = []
    num_labels = []
    i = 0
    for label in labels:
        j = 0
        while j < 10:
            real_labels.append(label)
            num_labels.append(i)
            j += 1
        i += 1
    print()
    import umap
    from sklearn.manifold import TSNE
    single_embedding = umap.UMAP(n_neighbors=5, min_dist=0.0, metric="cosine").fit_transform(img_feat)
    # single_embedding = TSNE(n_components=2, perplexity=20, n_iter=2000).fit_transform(img_feat)
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(single_embedding[:, 0], single_embedding[:, 1], c=num_labels, cmap='Spectral', alpha=1.0)
    plt.title('Single Modality Embedded via UMAP')
    plt.savefig("single.png")
    plt.close()

    multi_embedding = umap.UMAP(n_neighbors=5, min_dist=0.0, metric="cosine").fit_transform(fused_feat)
    # multi_embedding = TSNE(n_components=2, perplexity=20, n_iter=2000).fit_transform(fused_feat)

    fig_2, ax_2 = plt.subplots(1, figsize=(14, 10))
    plt.scatter(multi_embedding[:, 0], multi_embedding[:, 1], c=num_labels, cmap='Spectral', alpha=1.0)
    plt.title('Dual Modality Embedded via UMAP')
    plt.savefig("dual.png")
    plt.close()


if __name__ == '__main__':
    # main()
    plot_feat()

