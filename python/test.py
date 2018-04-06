import numpy as np
from PIL import Image
from scipy.special import expit


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def reconstruct_image_4():
    test_label = np.load("cache/test_predict2.npy")
    for image_index in range(len(test_label)):
        image = test_label[image_index].transpose(1, 2, 0)
        for x, i in enumerate(image):
            for y, j in enumerate(expit(i)):
                j = softmax(j)
                j0 = j[0]
                j1 = j[1]
                j2 = j[2]
                # print(j0,j1,j2)
                if abs(j0-j1) < 0.005 and abs(j0-j1) < 0.005:
                    image[x, y, 0] = 0
                    image[x, y, 1] = 0
                    image[x, y, 2] = 0
                    continue
                if j0 > j1 and j0 > j2:
                    image[x, y, 0] = 255
                else:
                    image[x, y, 0] = 0
                if j1 > j2 and j1 > j0:
                    image[x, y, 1] = 255
                else:
                    image[x, y, 1] = 0
                if j2 > j1 and j2 > j0:
                    image[x, y, 2] = 255
                else:
                    image[x, y, 2] = 0

        image[..., [1, 2]] = image[..., [2, 1]]
        # print(image.shape)
        # print(np.max(image, axis=2))
        # print(np.argmax(image, axis=2))
        image = Image.fromarray(np.uint8(image), mode='RGB')
        image.save('data/predict/label_{}.png'.format(image_index))
# image.show()

test_label = np.load("cache/test_predict2.npy")
for i in range(len(test_label)):
    image = test_label[i]
    image = np.uint8(np.where(image>= 0.1, 255, 0))
    Image.fromarray(image.reshape(565, 565), mode='L').show()
    exit()