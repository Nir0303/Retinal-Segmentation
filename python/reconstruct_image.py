import numpy as np
from PIL import Image


def build_image():
    test_label = np.load("cache/image/test_labels.npy")
    for i in range(len(test_label)):
        image = np.where(test_label[i]==True, 255, 0)
        image = Image.fromarray(np.uint8(image).transpose(1,2,0))
        image.save('data/predict/label_{}.png'.format(i))

build_image()