import numpy as np
import os
import re
from PIL import Image , ImageOps


pylon5 = os.environ["SCRATCH"] if os.environ.get("SCRATCH", None) else "."
DATA_PATH = os.path.join(pylon5, "data")


def get_image_path(data_type="train", image_type="label", dataset="big"):
    if dataset == "small":
        regex = re.compile(r"(.*_[1-8]\.png)|(.*\.tif)")
    else:
        regex = re.compile(r"(.*\.tif)")
    if data_type == "train" and image_type == "label":
        image_path = os.path.join(DATA_PATH, "train", "av")
    elif data_type == "train" and image_type == "image":
        image_path = os.path.join(DATA_PATH, "train", "images_1")
    elif data_type == "test" and image_type == "image":
        image_path = os.path.join(DATA_PATH, "test", "images_1")
    elif data_type == "test" and image_type == "label":
        image_path = os.path.join(DATA_PATH, "test", "av")

    for dir_path, sub_dirs, file_names in os.walk(image_path):
        for file_name in sorted(file_names):
            if regex.match(file_name):
                continue
            yield os.path.join(dir_path, file_name)


def load_images(data_type="train", image_type="label", classification=None, dataset="big"):
    images_data = []
    for index, image_path in enumerate(get_image_path(data_type, image_type, dataset)):
        image = Image.open(image_path)
        image_data = np.array(image, np.float32)
        if image_data.shape == (584, 565, 3):
            image_data = image_data[9:574,:,:]
        elif image_data.shape == (565, 584, 3):
            image_data = image_data[:,9:574, :]
        if image_type == "image":
            image_data -= np.array((171.0773, 98.4333, 58.8811))
        num_channels = len(image.getbands())
        image_data = image_data.reshape(image_data.shape[1], image_data.shape[0], num_channels)
        if image_type == "image":
            new_image_data = image_data.transpose((2, 0, 1))
        if image_type == "label":
            r = image_data[..., 0]
            g = image_data[..., 1]
            b = image_data[..., 2]
            new_image_data = np.stack([r,g,b], 0)
            unknown = (b+g+r <= 255)
            artery = (r > 0) & unknown
            vein = (b > 0) & unknown
            overlap = (g > 0) & unknown
            background = ((b + g + r) == 0)
            if classification == 3:
                # background =((b+g+r) == 0)
                new_image_data = np.stack([artery, vein, background], 0)
            elif classification == 4:
                new_image_data = np.stack([artery, overlap, vein, background], 0)
            elif classification == 1:
                optic_nerve = artery | overlap | vein
                new_image_data = np.stack([optic_nerve], 0)
            elif classification == 2:
                optic_nerve = artery | overlap | vein
                new_image_data = np.stack([optic_nerve,background], 0)
                # new_image_data = new_image_data[1,...].reshape(565,565)
                # Image.fromarray(np.uint8(np.where(new_image_data, 255, 0)), mode='L').show();exit();
            # Image.fromarray(np.uint8(np.where(new_image_data, 255, 0))).show()
        try:
            images_data.append(new_image_data)
        except Exception as e:
            print(e)
            # print(images_data.shape)
            # print(new_image_data.shape)

    images_data = np.array(images_data)
    return images_data


def load_drive_data(classification=3):
    train_images = load_images(data_type="train", image_type="image", classification=classification)
    train_labels = load_images(data_type="train", image_type="label", classification=classification)
    test_images = load_images(data_type="test", image_type="image", classification=classification)
    test_labels = load_images(data_type="test", image_type="label", classification=classification)

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    # load_drive_data()
    test_labels = load_images(data_type="test", image_type="label", classification=4)
    train_images, train_labels, test_images, test_labels = load_drive_data(classification=4)
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)
