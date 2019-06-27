import numpy as np

from PIL import Image

im = Image.open('../data/phone.jpg')
im = np.array(im)
print(im, im.dtype)
im = [255, 255, 255]-im
print(im)

Image.fromarray(im.astype(np.uint8)).save('../data/phone2.jpg')
