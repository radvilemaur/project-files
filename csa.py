import os
from natsort import natsorted
import cv2
import numpy as np


path = "/mnt/e/PROJECT/UNET/CSA"
i=0
for file in natsorted(os.listdir(path)):
 img = cv2.imread(file)
 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
 area2 = cv2.countNonZero(thresh)
 print(area2)
 i=i+1
