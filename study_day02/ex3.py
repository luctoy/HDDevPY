#
#    练习3：从图片中读取数据，处理写回图片
#
import numpy as np
from PIL import Image

im = Image.open('../data/phone.jpg')
im = np.array(im)
print(im.dtype)
im = [255,255,255] - im
print(im.dtype)
im = im.astype(np.uint8)
im = Image.fromarray(im)
im.save('../data/phone2.jpg')


