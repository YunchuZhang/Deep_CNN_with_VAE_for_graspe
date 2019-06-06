from __future__ import print_function
import torch
import torchvision

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint

root = '/home/yunchu/Workspace/Deep_CNN_with_VAE_for_graspe/dd/'

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
def gridiou(y_n,newlabel,batch):
	#print(y_n)

	ioutotal = torch.zeros(batch,newlabel.size(1),1)
	for i in range(batch):
		for j in range(newlabel.size(1)):
			box1 = y_n[i,j,:]
			box2 = newlabel[i,j,:]
			box1 = decode(box1,j)
			box2 = decode(box2,j)

			iou = caliou(box1,box2)
			ioutotal[i,j,:] = iou
	#print(ioutotal)

	return ioutotal

def decode(box,pos):

	ds = 80
	size = 640
	cx,cy,w,h,th = box[1:]
	ipos = pos//8
	jpos = pos - 8*(pos//8)

	cxt = jpos*ds + cx*ds
	cyt = ipos*ds + cy*ds
	wt = w*size
	ht = h*size

	box1 = ((cxt, cyt), (wt, ht), th)
	box1 = cv2.boxPoints(box1)
	#box1 = np.int0(box1)

	return box1

def change():
	with open(root + 'id.txt', 'r') as f:
		for line in f:
			line = line.rstrip()
			words = line.split()
			wordsa = words[0] + ".png"
			wordsb = words[0] + ".jpg"
			print(wordsa)
			img = cv2.imread(root+"{}".format(wordsa)) 
			#img = cv2.resize(img, dsize=(640, 640))
			cv2.imwrite("/home/yunchu/Workspace/Deep_CNN_with_VAE_for_graspe/a/"+"{}".format(wordsb), img)
			cv2.destroyAllWindows()



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

def drawresult(imgs,labels,dscale):
	de = dscale
	newlabel = labels
	newlabels = newlabel.numpy()
	print(newlabels.shape)
	for i in range(newlabels.shape[0]):
		print("Drawing %d img" %(i))
		img2 = imgs[i].numpy()*255
		img2 = img2.astype('uint8')
		img2 = np.transpose(img2, (1,2,0))
		img2=img2[:,:,::-1]#RGB->BGR
		cv2.imwrite("test"+"{}.png".format(i), img2)
		img = cv2.imread("test"+"{}.png".format(i)) 
		for each in newlabels[i]:
			box = each
			#print(box)
			a = box[0]/de
			b = box[1]/de
			c = box[2]/de
			d = box[3]/de
			e = box[4]
			box = ((a, b), (c, d), e)
			box = cv2.boxPoints(box)
			box = np.int0(box)
			#print(box)
			cv2.drawContours(img, [box], 0, (0, 255, 0), 1) 
		cv2.imwrite("test"+"{}.png".format(i), img)
		cv2.destroyAllWindows()

if __name__ == '__main__':
	change()









'''
	for r in range(batch):
		for c in range(64):

			if wp[r,c] < 0:
				
				lossw = lossw + (torch.sqrt(wl[r,c]) - torch.sqrt(-wp[r,c]))**2

			else:

				lossw = lossw + (torch.sqrt(wl[r,c]) - torch.sqrt(wp[r,c]))**2

			if hp[r,c] < 0:
				
				lossh = lossh + (torch.sqrt(hl[r,c]) - torch.sqrt(-hp[r,c]))**2

			else:

				lossh = lossh + (torch.sqrt(hl[r,c]) - torch.sqrt(hp[r,c]))**2
	lossw = lossw * lambda_coord
	lossh = lossh * lambda_coord
'''
'''
	ilist, jlist = np.where(newlabel[:, :, 0].detach().cpu().numpy() == 1)
	for r, c in zip(ilist, jlist):

		if wp[r,c] < 0:
			
			lossw = lossw + (torch.sqrt(wl[r,c]) + torch.sqrt(-wp[r,c]))**2

		else:

			lossw = lossw + (torch.sqrt(wl[r,c]) - torch.sqrt(wp[r,c]))**2

		if hp[r,c] < 0:
			
			lossh = lossh + (torch.sqrt(hl[r,c]) + torch.sqrt(-hp[r,c]))**2

		else:
			lossh = lossh + (torch.sqrt(hl[r,c]) - torch.sqrt(hp[r,c]))**2
'''