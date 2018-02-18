import numpy as np
import os
import re
from PIL import Image

DATA_PATH = os.path.join(os.getcwd(), "data")


def get_image_path(data_type="train", image_type="label"):
    regex = re.compile(r"(.*_[1-8]\.png)|(.*\.tif)")
    if data_type == "train" and image_type == "label":
        image_path = os.path.join(DATA_PATH, "train", "av")
    elif data_type == "train" and image_type == "image":
        image_path = os.path.join(DATA_PATH, "train", "images")
    elif data_type == "test" and image_type == "image":
        image_path = os.path.join(DATA_PATH, "test", "images")
    elif data_type == "test" and image_type == "label":
        image_path = os.path.join(DATA_PATH, "test", "av")

    for dir_path, sub_dirs, file_names in os.walk(image_path):
        for file_name in file_names:
            if regex.match(file_name):
                continue
            yield os.path.join(dir_path, file_name)


def load_images(data_type="train", image_type="label"):
    images_data = []
    for index, image_path in enumerate(get_image_path(data_type,image_type)):
        image = Image.open(image_path)
        image_data = np.array(image.getdata(), np.float32)
        num_channels = len(image.getbands())
        image_data = image_data.reshape(image.size[1], image.size[0], num_channels)

        if image_type == "image":
            new_image_data = image_data.transpose((2, 0, 1))
            #new_image_data = image_data.reshape(1, *image_data.shape)
        if image_type == "label":
            r = image_data[..., 0]
            g = image_data[..., 1]
            b = image_data[..., 2]
            unknown = (np.minimum(b, r)) > 0
            # background = ((b+g+r) == 0)
            artery = (r - unknown) > 0
            vein = (b - unknown) > 0
            overlap = (g - unknown) > 0
            new_image_data = np.stack([artery, vein, overlap], 0)
            new_image_data = new_image_data.astype(np.float32, copy=False)
            #print(new_image_data)
        try:
            images_data.append(new_image_data)
        except Exception as e:
            print(images_data.shape)
            print(new_image_data.shape)
    images_data = np.array(images_data)
    return images_data


def load_drive_data():

    train_images = load_images(data_type="train", image_type="image")
    train_labels = load_images(data_type="train", image_type="label")
    test_images = load_images(data_type="test", image_type="image")
    test_labels = load_images(data_type="test", image_type="label")

    return (train_images,train_labels,test_images,test_labels)


if __name__ == '__main__':
    load_drive_data()
