import pickle as pkl
import numpy as np
import os
from PIL import Image


def load_txt(file_path):
    with open(file_path) as f:
        all_cls_ids = f.readlines()
        all_cls_ids = [cls_id.strip() for cls_id in all_cls_ids]
    f.close()
    return all_cls_ids


def load_ids(file_path):
    with open(file_path) as f:
        all_cls_ids = f.readlines()
        all_cls_ids = [cls_id.strip() for cls_id in all_cls_ids]
    f.close()
    return all_cls_ids


def add_to_text_encode_dict(text, encode_dict):
    for sentence in text:
        encode_dict.append(sentence)


def build_encode_dict(encode_dict, output_root):
    ids = np.arange(len(encode_dict)).astype(np.str)
    id_sentence_encoded_dict = dict(zip(ids, encode_dict))
    sentence_id_encoded_dict = dict(zip(encode_dict, ids))

    with open(os.path.join(output_root, "id_sentence_encoder.pkl"), "wb") as f:
        pkl.dump(id_sentence_encoded_dict, f)
        f.close()

    with open(os.path.join(output_root, "sentence_id_encoder.pkl"), "wb") as f:
        pkl.dump(sentence_id_encoded_dict, f)
        f.close()


def main():
    root_path = "cvpr2016_cub"
    imgs_path = os.path.join(root_path, "images")
    texts_path = os.path.join(root_path, "text_c10")
    train_split_path = os.path.join(root_path, "trainids.txt")
    val_split_path = os.path.join(root_path, "valids.txt")
    test_split_path = os.path.join(root_path, "testids.txt")
    train_split = load_ids(train_split_path)
    val_split = load_ids(val_split_path)
    # test_split = load_split(test_split_path)
    text_encode_dict = []
    output_root = "pkl"
    os.makedirs(output_root, exist_ok=True)
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()
    # 200 classes
    all_imgs = os.listdir(imgs_path)
    img_class_list = []
    i = 1
    for img_class in all_imgs:
        img_class_path = os.path.join(imgs_path, img_class)
        if os.path.isdir(img_class_path):
            all_imgs_in_class = os.listdir(img_class_path)
            for img in all_imgs_in_class:
                if img.endswith("jpg"):
                    print(i)
                    text_name = img.split(".")[0] + ".txt"
                    img_class_list.append(img_class)
                    class_id, class_name = img_class.split(".")
                    full_img_path = os.path.join(img_class_path, img)
                    full_text_path = os.path.join(texts_path, img_class, text_name)
                    assert os.path.isfile(full_img_path) and os.path.isfile(full_text_path)

                    pil_image = Image.open(full_img_path)
                    pil_image = pil_image.resize((84, 84))
                    image = np.array(pil_image)
                    text = load_txt(full_text_path)
                    add_to_text_encode_dict(text, text_encode_dict)

                    if class_id in train_split:
                        if img_class in list(train_dict.keys()):
                            # train_dict[img_class].append([image, text])
                            train_dict[img_class]["img"].append(image)
                            train_dict[img_class]["text"].append(text)

                        else:
                            # train_dict[img_class] = [[image, text]]
                            train_dict[img_class] = dict()
                            train_dict[img_class]["img"] = []
                            train_dict[img_class]["text"] = []

                    elif class_id in val_split:
                        if img_class in list(val_dict.keys()):
                            # val_dict[img_class].append([image, text])
                            val_dict[img_class]["img"].append(image)
                            val_dict[img_class]["text"].append(text)
                        else:
                            # val_dict[img_class] = [[image, text]]
                            val_dict[img_class] = dict()
                            val_dict[img_class]["img"] = []
                            val_dict[img_class]["text"] = []
                    else:
                        if img_class in list(test_dict.keys()):
                            # test_dict[img_class].append([image, text])
                            test_dict[img_class]["img"].append(image)
                            test_dict[img_class]["text"].append(text)
                        else:
                            # test_dict[img_class] = [[image, text]]
                            test_dict[img_class] = dict()
                            test_dict[img_class]["img"] = []
                            test_dict[img_class]["text"] = []
                    i += 1

    data = dict()
    data["train"] = train_dict
    data["val"] = val_dict
    data["test"] = test_dict
    build_encode_dict(text_encode_dict, output_root)
    pkl_path = os.path.join(output_root, "data.pkl")
    with open(pkl_path, "wb") as f:
        pkl.dump(data, f)
        f.close()

    # train_pkl_path = os.path.join(output_root, "train_data.pkl")
    # with open(train_pkl_path, "wb") as f:
    #     pkl.dump(train_dict, f)
    #     f.close()
    #
    # val_pkl_path = os.path.join(output_root, "val_data.pkl")
    # with open(val_pkl_path, "wb") as f:
    #     pkl.dump(val_dict, f)
    #     f.close()
    #
    # test_pkl_path = os.path.join(output_root, "test_data.pkl")
    # with open(test_pkl_path, "wb") as f:
    #     pkl.dump(test_dict, f)
    #     f.close()


if __name__ == "__main__":
    main()
