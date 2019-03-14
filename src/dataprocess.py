from __future__ import print_function
import torch
import torchvision
#import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms
import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint
from img import imgshow

root = ''
def list2tensor(labels):
	label = torch.zeros([len(labels),labels[0].shape[0],4,2])
	for k in range(len(labels)):
		for j in range(labels[0].shape[0]):
			label[k,j,:,:] = labels[k][j]
	label = torch.transpose(label,0,1)
	#print(label)
	#print(label.shape)
	return(label)


def makeid():
	with open("id.txt","w") as f:
		for i in range(100):
			a = "pcd0"+"{}".format(100 + i)
			f.write(a)
			f.write('\n')

def boxtolabel(labeltxt):
	a = np.loadtxt(labeltxt)
	boxes = []
	box = []
	#print(a.shape[0])
	for i in range(a.shape[0]):
		if i%4 ==0:
			print("Processing %d box" %(i/4+1))
			box = [a[i],a[i+1],a[i+2],a[i+3]]
			box = np.int0(box)
			boxes.append(box)
			print(box)
	return(boxes)

#"pcd0"+"{}".format(100 + i) + "r.png"
#"pcd0"+"{}".format(100 + i) + "cpos.txt"

class MyDataset(torch.utils.data.Dataset): 
	def __init__(self,root, datatxt, transform=None, target_transform=None):
		imgs = [] 
		with open(root + datatxt, 'r') as f:
			for line in f:
				line = line.rstrip()
				words = line.split()
				wordsa = words[0] + "r.png"
				wordsb = words[0] + "cpos.txt"
				print("Loading %s" %(wordsa))
				#img label
				imgs.append((wordsa,wordsb))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
	def __getitem__(self, index):
		fn, labeltxt = self.imgs[index]
		img = Image.open(root+fn).convert('RGB')
		label = boxtolabel(labeltxt)
		if self.transform is not None:
			img = self.transform(img)
		return img,label
	def __len__(self):
		return len(self.imgs)
def test():
	train_data=MyDataset(root=root,datatxt ='id.txt', transform=transforms.ToTensor())
#test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
	train_loader = Data.DataLoader(dataset=train_data, batch_size=96, shuffle=False)
	for i, data in enumerate(train_loader):
		imgs, labels= data
		if i==1:
			print(labels)
			newlabel = list2tensor(labels)
			print(newlabel)
			print(newlabel[0,:])
			img = transforms.ToPILImage()(imgs[0])
			#img.show()
			#imgshow(imgs[0])
			

if __name__ == '__main__':
	test()

