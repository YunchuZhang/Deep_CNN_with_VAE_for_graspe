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
from img import imgshow,drawresult,gridiou,decode

grid_size = (8,8,6)
dscale = 640/8
sizet = 640
root = '/home/yunchu/Workspace/Deep_CNN_with_VAE_for_graspe/Jacquard_Dataset_Merged/'

def list2tensor(labels):
	label = torch.zeros([len(labels),labels[0].shape[0],5])
	for k in range(len(labels)):
		for j in range(labels[0].shape[0]):
			label[k,j,:] = labels[k][j]
	label = torch.transpose(label,0,1)
	#print(label)
	#print(label.shape)
	return(label)


def makeid():
	with open("id.txt","w") as f:
		for i in range(1400):
			a = "{}".format(4001 + i)
			f.write(a)
			f.write('\n')

def boxtolabel(labeltxt):
	boxes = []
	box = []
	i = 0
	#j = 0
	with open(labeltxt, 'r') as f:
		for line in f:
			line = line.rstrip()
			a = line.split(';')
			#print("Processing %d box" %(i))
			# delete angel > 90 
			# delete repeated 
			if float(a[2])<0:
				# x y th open jawsize
				# x y th w h
				box = [float(a[0]),float(a[1]),float(a[3]),float(a[4]),float(a[2])]
			else:
				# x y th h w
				box = [float(a[0]),float(a[1]),float(a[4]),float(a[3]),float(a[2])]
			#box = np.int0(box)
			boxes.append(box)

			i = i + 1

				#print(box)
				
	#print("Org_boxes %d"%(i))
	#print("Fil_boxes %d"%(j))
	
	return(boxes)

#"pcd0"+"{}".format(100 + i) + "r.png"
#"pcd0"+"{}".format(100 + i) + "cpos.txt"

def getgridlabel(labels,grid_size = (8,8,6),name):
	name = int(name)
	bboxnew = torch.zeros(grid_size)
	total = 0
	a = np.asarray([0,0,0,0,0,0,0,0])
	for k in range(len(labels)):

		cx, cy, w, h, th = labels[k]
		cxt = 1.0 * cx / 1024
		cyt = 1.0 * cy / 1024
		wt = 1.0 * w / 1024
		ht = 1.0 * h / 1024
		#print(wt)
		#print(ht)

		j = int(np.floor(cxt / (1.0 / grid_size[0])))
		i = int(np.floor(cyt / (1.0 / grid_size[1])))
		xn = (cxt * sizet - j * dscale) / dscale
		yn = (cyt * sizet - i * dscale) / dscale

		#print ("one box is {}".format((i, j, xn, yn, wt, ht)))
		label_save = np.asarray([i, j, 1, xn, yn, wt, ht,th])
		label_save= np.vstack((a,label_save))
		label_vec = np.asarray([1, xn, yn, wt, ht,th])
		label_vec = torch.from_numpy(label_vec)
		#print ("Final box is {}".format(label_vec))
		bboxnew[i,j,:] = label_vec
	print(bboxnew)
	np.savetxt("{}".format(name)+".txt", label_save)

	return(bboxnew)


class MyDataset(torch.utils.data.Dataset): 
	def __init__(self,root, datatxt, transform=None, target_transform=None):
		imgs = [] 
		with open(root + datatxt, 'r') as f:
			for line in f:
				line = line.rstrip()
				words = line.split()
				wordsa = words[0] + ".jpg"
				wordsb = words[0] + ".txt"
				print("Loading %s" %(wordsa))
				#img label
				imgs.append((wordsa,wordsb，words[0]))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
	def __getitem__(self, index):
		fn, labeltxt,name = self.imgs[index]
		img = Image.open(root+fn).convert('RGB')
		#print("Processing %s box" %(labeltxt))
		label = boxtolabel(root+labeltxt)
		label = getgridlabel(label,grid_size,name)

		if self.transform is not None:
			img = self.transform(img)
		return img,label
	def __len__(self):
		return len(self.imgs)

def test():
	train_data=MyDataset(root=root,datatxt ='id.txt',transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
#test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
	train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=False)
	for i, data in enumerate(train_loader):
		imgs, labels= data
		if i==0:
			#print(labels)
			#newlabel = list2tensor(labels)
			print(labels.size())
			#drawresult(imgs,newlabel,dscale = 1024/640.0)
			#img = transforms.ToPILImage()(imgs[0])
			#img.show()
			#imgshow(imgs[0])

if __name__ == '__main__':
	test()
	#makeid()