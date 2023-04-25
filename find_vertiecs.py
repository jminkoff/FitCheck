# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:12:18 2023

@author: jade

"""
##from PIL import Image
import cv2
import numpy as np

# create red image #replace with image resized from other script
img = np.full((10,10,3), (0,0,255), dtype=np.uint8)

# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# get coordinates (y,x) --- alternately see below for (x,y)
yx_coords = np.column_stack(np.where(gray >= 0))
print (yx_coords)

print ('')

# get coordinates (x,y)
xy_coords = np.flip(np.column_stack(np.where(gray >= 0)), axis=1)
print (xy_coords)


##image = Image.open(img)
##image.show()
"""
from PIL import Image, ImageDraw, ImageFont
image = Image.open("image.png")
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 20, encoding="unic")
draw.text( (10,10), u"Your Text", fill=‘#a00000’, font=font)
image.save("out.png","PNG")
"""