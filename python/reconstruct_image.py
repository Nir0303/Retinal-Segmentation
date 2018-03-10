import numpy as np
from PIL import Image

from scipy.special import expit


def build_image():
    test_label = np.load("cache/test_predict.npy")
    for i in range(len(test_label)):
        #print(test_label[i])
        x = test_label[i]
        image = np.where(x >= 0.2, 255, 0)
        image = Image.fromarray(np.uint8(image).transpose(1,2,0))
        image.save('data/predict/label_{}.png'.format(i))

build_image()