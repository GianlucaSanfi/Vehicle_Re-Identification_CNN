from PIL import Image
import os

#create 10 test images (useless images)
for i in range(1, 11):
    img = Image.new('RGB', (128, 256), color = (i*10, i*20, i*15))
    img.save(f'datasets/TEST/images/vehicle_{i}.jpg')

