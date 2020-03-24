import numpy as np
import imageio
from scipy import misc
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff
import os


path = "/mnt/e/PROJECT/UNET/RESULTS/dice_lr0.001"
image_number=30
JI=0
Hausdorff=0
DL=0

def dice_coefficient(y_true, y_pred):
  return 2 * np.sum(y_true * y_pred)/(np.sum(y_true) + np.sum(y_pred))

for i in range(image_number):
 im1 = imageio.imread(os.path.join(path, '%d_binary.png'%i))
 im2 = imageio.imread(os.path.join(path, '%d_predict.png'%i))
 y_true = np.asarray(im1).astype(np.bool)
 y_pred = np.asarray(im2).astype(np.bool)
 jaccard= jaccard_score(y_true, y_pred, average='micro')
 print(jaccard)
 HD=directed_hausdorff(y_true, y_pred)
 print(HD[0])
 dice=dice_coefficient(y_true,y_pred)
 print(dice)
 JI= JI + jaccard
 Hausdorff = Hausdorff + HD[0]
 DL= DL + dice



averageJI= JI / image_number
print('Average Jaccard Index:', averageJI)
averageHD= Hausdorff / image_number
print('Average HD:', averageHD)
averageDL= DL / image_number
print("Averge Dice score:", averageDL)
