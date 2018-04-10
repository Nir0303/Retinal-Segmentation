import numpy as np
from PIL import Image
from scipy.special import expit
from PIL import ImageDraw

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def reconstruct_image(classification='4_class', data=None):
    if not data:
        return
    predict_label = np.load(data)
    for image_index in range(len(predict_label)):
        image_data = predict_label[image_index]
        image_data = softmax(expit(image_data)).transpose(1, 2, 0)
        # image_data = softmax(expit(image_data))
        for x, i in enumerate(image_data):
            for y, j in enumerate(i):
                j = softmax(expit(j))
                if classification == '2_class':
                    j0 = j[0]
                    j1 = j[1]
                    if j0 > j1:
                        image_data[x, y, 0] = 255
                    else:
                        image_data[x, y, 0] = 0
                elif classification == '4_class':
                    j0 = j[0]
                    j1 = j[1]
                    j2 = j[2]
                    j3 = j[3]
                    # print(j0, j1, j2, j3)
                    if j0 >= j1 and j0 >= j2 and j0 >= j3:
                        image_data[x, y, 0] = 255
                    else:
                        image_data[x, y, 0] = 0
                    if j1 >= j0 and j1 >= j2 and j1 >= j3:
                        image_data[x, y, 2] = 255
                    else:
                        image_data[x, y, 2] = 0
                    if j2 >= j1 and j2 >= j1 and j2 >= j3:
                        image_data[x, y, 1] = 255
                    else:
                        image_data[x, y, 1] = 0
                    if j3 >= j0 and j3 >= j2 and j3 >= j1:
                        image_data[x, y, 0] = 0
                        image_data[x, y, 1] = 0
                        image_data[x, y, 2] = 0
                    image_data = image_data[..., 0:3]
            
            if classification == '2_class':
                image = Image.fromarray(np.uint8(image_data[..., 0]).reshape(565, 565), mode='L')
            elif classification == '4_class':
                image = Image.fromarray(np.uint8(image_data), mode='RGB')
            image.save('data/predict/{}/label_{}.png'.format(classification,image_index))


# reconstruct_image(classification='2_class', data="cache/test_predict2_class_2.npy")
reconstruct_image(classification='4_class', data="cache/test_predict2_class_4_1.npy")