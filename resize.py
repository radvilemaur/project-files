import numpy as np
import imageio
from PIL import Image
import os
import cv2
num_image=30
path = "/mnt/e/PROJECT/UNET/RESULTS/dice_lr0.001/resize"
for i in range(num_image):
 im1= Image.open(os.path.join(path,'%d.png'%i))
 new_width=256
 new_height=256
 im1_resized=im1.resize((new_width, new_height), Image.ANTIALIAS)
 im1_resized.save('%d_resized.png'%i) #save resized image
#read image as grey scale
 img_grey = cv2.imread('%d_resized.png'%i, cv2.IMREAD_GRAYSCALE)
#assign blue channel to zeros
 img_binary = cv2.threshold(img_grey, 100, 255, cv2.THRESH_BINARY)[1]
 #save image
 cv2.imwrite('%d_binary.png'%i,img_binary)


#code to check ALL pixel values of an image
# from PIL import Image
# im=Image.open('1.png','r')
# pix_val=list(im.getdata())
# print(pix_val)
