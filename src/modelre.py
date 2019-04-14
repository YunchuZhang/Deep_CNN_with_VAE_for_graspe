import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from dataprocess2 import *
from torchvision import models
import time
import copy
from tensorboardX import SummaryWriter


writer = SummaryWriter(log_dir='scalar')


#dataseg 700 175 batch 25
#580 260 45
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss(size_average = False)
EPOCH = 300
BATCH_SIZE = 200
#0.001 0.0003
LR = 0.0035
GPU = True
train_size = 20000
val_size = 14000
tesize = 465

conf_threshold = 0.75

# id1 train id2 val id3 test
# 
train_data=MyDataset(root=root,datatxt ='id1.txt',transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

val_data=MyDataset(root=root,datatxt ='id2.txt', transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)

test_data=MyDataset(root=root,datatxt ='id3.txt', transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
test_loader = Data.DataLoader(dataset=test_data, batch_size=465, shuffle=False)

# # !!!!!!!! Change in here !!!!!!!!! #
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
# test_y = test_data.test_labels[:2000].cuda()


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
								   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
		self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
		self.out = nn.Linear(32 * 7 * 7, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		output = self.out(x)
		return output
def totalloss(y,label,batch):
	# Reshape input to desired shape
	newlabel = torch.reshape(label,(batch,64,6))
	y_n = torch.reshape(y,(batch,64,6))

	# split the tensor in the order of [prob, x, y, w, h, y, theta]
	cl, xl, yl, wl, hl, thl = torch.split(newlabel, 1, dim=2)
	cp, xp, yp, wp, hp, thp = torch.split(y_n, 1, dim=2)

	# weight different target differently
	lambda_coord = 5
	lambda_objc = 12
	lambda_obj = 1
	lambda_noobj = 0.3
	mask = cl*lambda_obj+(1-cl)*lambda_noobj
	iou = gridiou(y_n,newlabel,batch)
	# linear regression
	lossc = mse(cl*mask*lambda_objc, cp*mask*lambda_objc)
	lossiou = mse(cl*mask, cp*mask*iou)
	lossx = mse(xl*cl*lambda_coord, xp*cp*lambda_coord)
	lossy = mse(yl*cl*lambda_coord, yp*cp*lambda_coord)
	lossw = mse(tensor.sqrt(wl)*cl*lambda_coord, tensor.sqrt(wp)*cp*lambda_coord)
	lossh = mse(tensor.sqrt(hl)*cl*lambda_coord, tensor.sqrt(hp)*cp*lambda_coord)
	lossth = mse(thl*cl*lambda_coord, thp*cp*lambda_coord)
	
	# return joint loss
	loss = lossc+lossiou+lossx+lossy+lossw+lossh+lossth
	return loss

def accmetrix(y,label,batch):

	newlabel = torch.reshape(label,(batch,64,6))
	y_n = torch.reshape(y,(batch,64,6))


	c_label = newlabel[:, :, 0]
	c_pred = y_n[:, :, 0]
	boxes_pred = c_pred > conf_threshold
	sum_tp = np.sum(c_label * boxes_pred)
	sum_tn = np.sum((1 - c_label) * (1 - boxes_pred))
	sum_fn = np.sum(c_label * (1 - boxes_pred))
	sum_fp = np.sum(boxes_pred * (1 - c_label))

	num_boxes = np.sum(c_label)
	sum_conf = np.sum(np.abs(c_pred - c_label)) / (64 * batch)
	### delete c_label for 0 1
	sum_x = np.sum((np.abs(y_n[:, :, 1] - newlabel[:, :, 1])) * c_label) * 80 / num_boxes
	sum_y = np.sum((np.abs(y_n[:, :, 2] - newlabel[:, :, 2])) * c_label) * 80 / num_boxes
	sum_w = np.sum(np.abs(y_n[:, :, 3] - newlabel[:, :, 3]) * c_label) * 640/ num_boxes
	sum_h = np.sum(np.abs(y_n[:, :, 4] - newlabel[:, :, 4]) * c_label) * 640 / num_boxes
	sum_th = np.sum(np.abs(y_n[:, :, 5] - newlabel[:, :, 5]) * c_label)  / num_boxes

	c_accuracy = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)
	c_precision = sum_tp / (sum_tp + sum_fp + 1e-6)
	c_recall = sum_tp / (sum_tp + sum_fn + 1e-6)
	x_diff = sum_x
	y_diff = sum_y
	w_diff = sum_w
	h_diff = sum_h
	th_diff = sum_th

	#print(total)
	return(c_accuracy,c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff)


def train():
	#build model
	##input batch*3*480*640
	##output batch*25*4*2  batch*200
	#
	since = time.time()

	best_acc = 0.0
	

	
	
	model = models.resnet34(pretrained = False)
	fc_features = model.fc.in_features
	model.fc = nn.Linear(fc_features,384)

	best_model_wts = copy.deepcopy(model.state_dict())

	if (GPU):
		model = model.to(device)


	optimizer = optim.Adam(model.parameters(), lr=LR)
	# Decay LR by a factor of 0.1 every 7 epochs
	scheduler = lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.9)

	for epoch in range(EPOCH):
		print('Epoch {}/{}'.format(epoch, EPOCH - 1))
		print('-' * 25)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			running_loss = 0.0
			running_acc = 0

			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
				loader = train_loader
				tsize = train_size

			else:
				model.eval()   # Set model to evaluate mode
				loader = val_loader
				tsize = val_size

			for i, data in enumerate(loader):
				x, label= data
				
				if (GPU):
					x = x.to(device)
					label = label.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					y = model(x)

					loss = totalloss(y, label , x.size(0))

					if phase == 'train':
						loss.backward()
						optimizer.step()
				# statistics
				running_loss += loss.item()

				c_accuracy,c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff = accmetrix(y, label , x.size(0))

				running_acc += c_accuracy

				batch_loss = loss.item()
				batch_acc = c_accuracy

				print('*' * 20)
				print('{} Batch_Loss: {:.4f} Batch_Acc: {:.4f}'.format(phase, batch_loss, batch_acc))
				print('{} Batch_precision: {:.4f} Batch_recall: {:.4f} Batch_x_diff: {:.4f} \
					Batch_y_diff: {:.4f} Batch_w_diff: {:.4f} Batch_h_diff: {:.4f} Batch_th_diff: {:.4f}\
					'.format(phase, c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff))
				print('*' * 20)

			epoch_loss = running_loss / tsize
			epoch_acc = running_acc / tsize
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase =='train':
				writer.add_scalars('scalar/scalars_train', {'trainloss': epoch_loss, 'trainacc': epoch_acc}, epoch)

			else :
				writer.add_scalars('scalar/scalars_val', {'valloss': epoch_loss, 'valacc': epoch_acc}, epoch)
				#writer.add_scalars('scalar/scalars_test', {'xsinx': epoch * np.sin(epoch), 'xcosx': epoch * np.cos(epoch)}, epoch)

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
	



	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	torch.save(model.state_dict(),'model_para.pkl')
	writer.close()

	return model

def test(model):


	was_training = model.training
	model.eval()

	running_acc = 0.0

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			x, label= data
			
			if (GPU):
				x = x.to(device)
				label = label.to(device)

			y = model(x)
			#c_accuracy,c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff = accmetrix(y, label , x.size(0))
			running_acc += accmetrix(y, label , x.size(0))[0]
		epoch_acc = running_acc / tesize
		print('-' * 20)
		print(' Test Acc: {:.4f}'.format(epoch_acc))
		model.train(mode=was_training)



if __name__ == '__main__':
	#model = models.resnet18(pretrained = False)
	#fc_features = model.fc.in_features
	#model.fc = nn.Linear(fc_features,24)
	#model.load_state_dict(torch.load('model_para0.05.pkl'))
	model = train()
	#net.load_state_dict(torch.load('model_para.pkl'))
	test(model)

