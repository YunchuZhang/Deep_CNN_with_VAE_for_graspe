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
#from img import imgshow

root = ''
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
		for i in range(15):
			a = "{}".format(0 + i)
			f.write(a)
			f.write('\n')

def boxtolabel(labeltxt):
	boxes = []
	box = []
	i = 0
	j = 0
	with open(labeltxt, 'r') as f:
		for line in f:
			line = line.rstrip()
			a = line.split(';')
			#print("Processing %d box" %(i))
			# delete angel > 90 
			# delete repeated 
			if float(a[4])<=90:
				box = [float(a[0]),float(a[1]),float(a[2]),float(a[3]),-float(a[4])]
				box = np.int0(box)
				if i == 0:
					boxes.append(box)
					j = j + 1
				else:
					if boxes[-1][0] == box[0] and boxes[-1][1] == box[1] and boxes[-1][2]== box[2] and boxes[-1][3]== box[3]:
						pass
					else:
						boxes.append(box)
						j = j + 1

				i = i + 1

				#print(box)
				
	print("Org_boxes %d"%(i))
	print("Fil_boxes %d"%(j))
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
				wordsa = words[0] + ".png"
				wordsb = words[0] + ".txt"
				print("Loading %s" %(wordsa))
				#img label
				imgs.append((wordsa,wordsb))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
	def __getitem__(self, index):
		fn, labeltxt = self.imgs[index]
		img = Image.open(root+fn).convert('RGB')
		print("Processing %s box" %(labeltxt))
		label = boxtolabel(root+labeltxt)
		if self.transform is not None:
			img = self.transform(img)
		return img,label
	def __len__(self):
		return len(self.imgs)
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

def test():
	train_data=MyDataset(root=root,datatxt ='id.txt',transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
#test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
	train_loader = Data.DataLoader(dataset=train_data, batch_size=15, shuffle=False)
	for i, data in enumerate(train_loader):
		imgs, labels= data
		if i==0:
			#print(labels)
			newlabel = list2tensor(labels)
			print(newlabel.size())
			drawresult(imgs,newlabel,dscale = 1024/640.0)
			#img = transforms.ToPILImage()(imgs[0])
			#img.show()
			#imgshow(imgs[0])

if __name__ == '__main__':
	test()
	#makeid()

