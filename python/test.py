import numpy as np
from PIL import Image
from scipy.special import expit
from PIL import ImageDraw

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def reconstruct_image_3():
    test_label = np.load("cache/test_predict2_3class.npy")
    for image_index in range(len(test_label)):
        image = test_label[image_index].transpose(1, 2, 0)
        for x, i in enumerate(image):
            for y, j in enumerate(expit(i)):
                j = softmax(j)
                j0 = j[0]
                j1 = j[1]
                j2 = j[2]
                # print(j0,j1,j2)

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

def reconstruct_1class():
    test_label = np.load("cache/test_predict2.npy")
    for i in range(len(test_label)):
        print("yes")
        image_data = test_label[i]
        image_data = np.uint8(np.where(image_data>= 0.00001, 255, 0)).reshape(565, 565)
        image = Image.fromarray(image_data, mode='L')
        image.save('data/predict/1_class/label_{}.png'.format(i+1))


# reconstruct_image_3()
reconstruct_1class()