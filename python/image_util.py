import os
import numpy as np
from PIL import ImageDraw
from PIL import Image

"""
print("yes")
img = Image.new('RGB', (565, 565))
drw = ImageDraw.Draw(img, 'RGBA')
drw.ellipse((10, 10, 560, 550), fill=(255,255,255))
# drw.polygon([(50,100), (100, 0), (0, 0)], (0, 255, 0, 125))
del drw
img.save('circle_2.png', 'PNG')
"""

def crop_circle():
    image_path = 'data/train/images/'
    for dir_path, sub_dirs, file_names in os.walk(image_path):
        for file_name in sorted(file_names):
            if file_name.endswith('.tif'):
                continue
            image_data = np.array(Image.open(os.path.join(dir_path, file_name)))
            if image_data.shape == (584, 565, 3):
                image_data = image_data[9:574, :, :]
                im = np.array(Image.open('cache/circle_1.png'))
            else:
                image_data = image_data[:, 9:574, :]
                im = np.array(Image.open('cache/circle_2.png'))

            for i in range(0, 565):
                for j in range(0, 565):
                    if im[i, j, 0] == 0 and im[i, j, 1] == 0 and im[i, j, 2] == 0:
                        image_data[i, j, :] = 0
            image = Image.fromarray(np.uint8(image_data), mode='RGB')
            image.save('data/train/images_1/{}'.format(file_name))

crop_circle()
