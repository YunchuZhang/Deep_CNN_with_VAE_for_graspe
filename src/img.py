from __future__ import print_function
import torch
import torchvision

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint



def caliou(box1,box2):
	poly1 = Polygon(box1).convex_hull  #四个点顺序为：左上 左下  右下 右上 左上
	#print(poly1)
	poly2 = Polygon(box2).convex_hull  
	#print(poly2)
	if not poly1.intersects(poly2): #如果两四边形不相交
		iou = 0.0
	else:
		iou = poly1.intersection(poly2).area / poly1.union(poly2).area
	#print(iou)
	return(iou)

def imgshowbox(imgfile):
	img = cv2.imread("{}.png".format(imgfile)) 
	cv2.namedWindow("Image") 
	for eachbox in boxes:
		box = eachbox
		cv2.drawContours(img, [box], 0, (0, 255, 0), 1) 
	#cv2.imshow('Image', img)
	#cv2.waitKey (0)
	cv2.imwrite("test.png", img)
	cv2.destroyAllWindows()
def imgshow(img):
    img2 = img.numpy()*255
    img2 = img2.astype('uint8')
    img2 = np.transpose(img2, (1,2,0))
    img2=img2[:,:,::-1]#RGB->BGR
    cv2.imshow('img2', img2)
    cv2.waitKey()