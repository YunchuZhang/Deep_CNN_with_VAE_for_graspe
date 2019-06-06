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
EPOCH = 80
BATCH_SIZE = 20
#0.0035 0.035
# 0.01 640 png 25epoch 134756
# 0.01 640 jpg res34 1 29****29 ***
# 0.003 640 jpg resnet 18 104epoch 30891
# 0.003 640jpg res-18 80epoch 10012 loss
# 0.0025 640jpg res-18 114 3780loss val 166000
# 0.0012 640jpg res-18 72 214683 loss val 954813
LR = 0.0008
GPU = True
train_size = 35000
val_size = 15000
tesize = 465

conf_threshold = 0.75

# id1 train id2 val id3 test
# 
'''
train_data=MyDataset(root=root,datatxt ='id1.txt',transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)


val_data=MyDataset(root=root,datatxt ='id2.txt', transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)

test_data=MyDataset(root=root,datatxt ='id3.txt', transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
test_loader = Data.DataLoader(dataset=test_data, batch_size=465, shuffle=False)
'''
train_data=MyDataset(root=root,datatxt ='id1.txt', transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data=MyDataset(root=root,datatxt ='id2.txt', transform=transforms.ToTensor())
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)


test_data=MyDataset(root=root,datatxt ='id3.txt', transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# # !!!!!!!! Change in here !!!!!!!!! #
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
# test_y = test_data.test_labels[:2000].cuda()

class MyModel(nn.Module):
	def __init__(self, pretrained_model):
		super(MyModel, self).__init__()
		self.pretrained_model = pretrained_model
		self.last_layer = nn.Sigmoid()

	def forward(self, x):
		return self.last_layer(self.pretrained_model(x))




def totalloss(y,label,batch):
	# Reshape input to desired shape
	newlabel = torch.reshape(label,(batch,64,6))
	y_n = torch.reshape(y,(batch,64,6))

	# split the tensor in the order of [prob, x, y, w, h, y, theta]
	cl, xl, yl, wl, hl, thl = torch.split(newlabel, 1, dim=2)
	cp, xp, yp, wp, hp, thp = torch.split(y_n, 1, dim=2)
	cp = 1 / (1 + torch.exp(-cp))
	xp = 1 / (1 + torch.exp(-xp))
	yp = 1 / (1 + torch.exp(-yp))
	wp = 1 / (1 + torch.exp(-wp))
	hp = 1 / (1 + torch.exp(-hp))
	#thp = (thp - 0.5) * 180

	# weight different target differently
	lambda_coord = 6
	lambda_objc = 10
	lambda_obj = 1
	lambda_noobj = 0.2
	mask = cl*lambda_obj+(1-cl)*lambda_noobj
	#iou = gridiou(y_n,newlabel,batch)
	#iou = iou.to(device)
	# linear regression
	lossc = mse(cl*mask*lambda_objc, cp*mask*lambda_objc)
	#lossiou = mse(cl*mask, cl*mask*iou)
	lossx = mse(xl*cl*lambda_coord, xp*cl*lambda_coord)
	lossy = mse(yl*cl*lambda_coord, yp*cl*lambda_coord)


	lossw = mse(torch.sqrt(wl)*cl*lambda_coord, torch.sqrt(wp)*cl*lambda_coord)
	lossh = mse(torch.sqrt(hl)*cl*lambda_coord, torch.sqrt(hp)*cl*lambda_coord)


	#print(thp)
	#print(thl)
	#print(thp)
	#print(torch.sum(thl))
	#print(torch.sum(thp*cl))
	thl = thl*cl
	thp = thp*cl
	#(mse(thl, thp))

	lossth = mse(thl*(lambda_coord), thp*(lambda_coord))
	
	# return joint loss
	loss = lossc+lossx+lossy+lossw+lossh+lossth

	print(lossc.item(),lossx.item(),lossy.item(),lossw.item(),lossh.item(),lossth.item())
	return loss

def accmetrix(y,label,batch):

	newlabel = torch.reshape(label,(batch,64,6))
	y_n = torch.reshape(y,(batch,64,6))

	y_n[:, :, 0:5] = 1 / (1 + torch.exp(-y_n[:, :, 0:5]))
	c_label = newlabel[:, :, 0]
	c_pred = y_n[:, :, 0]
	#print(c_label,c_pred)
	boxes_pred = c_pred > conf_threshold
	boxes_pred = boxes_pred.float()
	sum_tp = torch.sum(c_label * boxes_pred)
	sum_tn = torch.sum((1 - c_label) * (1 - boxes_pred))
	sum_fn = torch.sum(c_label * (1 - boxes_pred))
	sum_fp = torch.sum(boxes_pred * (1 - c_label))

	num_boxes = torch.sum(c_label)
	sum_conf = torch.sum(torch.abs(c_pred - c_label)) / (64 * batch)
	### delete c_label for 0 1
	sum_x = torch.sum(torch.abs(y_n[:, :, 1] * c_label - newlabel[:, :, 1])) * 80 / num_boxes
	sum_y = torch.sum(torch.abs(y_n[:, :, 2] * c_label - newlabel[:, :, 2])) * 80 / num_boxes
	sum_w = torch.sum(torch.abs(y_n[:, :, 3] * c_label - newlabel[:, :, 3])) * 640/ num_boxes
	sum_h = torch.sum(torch.abs(y_n[:, :, 4] * c_label - newlabel[:, :, 4])) * 640 / num_boxes
	#print(y_n[:, :, 5] )
	#thp = ((y_n[:, :, 5]) - 0.5) * 180
	#print(thp)
	#print(thp  * c_label)
	sum_th = torch.sum(torch.abs(y_n[:, :, 5]  * c_label - newlabel[:, :, 5]))  / num_boxes

	c_accuracy = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)
	#print(sum_tp,sum_tn,sum_fp,sum_fn)
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
	

	modelpre = models.resnet34(pretrained = False)
	fc_features = modelpre.fc.in_features
	modelpre.fc = nn.Linear(fc_features,384)

	#model = MyModel(modelpre)
	model = modelpre
	


	best_model_wts = copy.deepcopy(model.state_dict())

	if (GPU):
		model = model.to(device)


	optimizer = optim.Adam(model.parameters(), lr=LR , weight_decay = 0.15)
	# Decay LR by a factor of 0.1 every 7 epochs
	scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

	for epoch in range(EPOCH):
		print('Epoch {}/{}'.format(epoch, EPOCH - 1))
		print('-' * 25)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			running_loss = 0.0
			running_acc = 0
			epoch_acc = 0.0
			epoch_x = 0.0
			epoch_y = 0.0
			epoch_w = 0.0
			epoch_h =  0.0
			epoch_th = 0.0

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
				epoch_acc += batch_acc * x.size(0)
				epoch_x += x_diff
				epoch_y += y_diff
				epoch_w += w_diff
				epoch_h += h_diff
				epoch_th += th_diff		


				#print('*' * 20)
				if i%10 == 0:
					print('{} Batch :{} Batch_Loss: {:.4f} Batch_Acc: {:.4f}'.format(i, phase, batch_loss, batch_acc))
					print('{} Batch_precision: {:.4f} Batch_recall: {:.4f} Batch_x_diff: {:.4f} Batch_y_diff: {:.4f} Batch_w_diff: {:.4f} Batch_h_diff: {:.4f} Batch_th_diff: {:.4f}\
					'.format(phase, c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff))
					print('*' * 20)

			epoch_loss = running_loss / tsize
			epoch_acc = epoch_acc / tsize
			epoch_x = epoch_x / (tsize/BATCH_SIZE)
			epoch_y = epoch_y / (tsize/BATCH_SIZE)
			epoch_w = epoch_w / (tsize/BATCH_SIZE)
			epoch_h = epoch_h / (tsize/BATCH_SIZE)
			epoch_th = epoch_th / (tsize/BATCH_SIZE)


			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase =='train':
				writer.add_scalars('scalar/scalars_tloss', {'trainloss': epoch_loss}, epoch)
				writer.add_scalars('scalar/scalars_tacc', { 'trainacc': epoch_acc}, epoch)
				writer.add_scalars('scalar/scalars_txy', {'trainx': epoch_x, 'trainy': epoch_y}, epoch)
				writer.add_scalars('scalar/scalars_twh', {'trainw': epoch_w, 'trainh': epoch_h}, epoch)
				writer.add_scalars('scalar/scalars_tth', {'trainth': epoch_th}, epoch)

			else :
				writer.add_scalars('scalar/scalars_vloss', {'valloss': epoch_loss}, epoch)
				writer.add_scalars('scalar/scalars_vacc', {'valacc': epoch_acc}, epoch)
				writer.add_scalars('scalar/scalars_vxy', {'valx': epoch_x, 'valy': epoch_y}, epoch)
				writer.add_scalars('scalar/scalars_vwh', {'valw': epoch_w, 'valh': epoch_h}, epoch)
				writer.add_scalars('scalar/scalars_vth', {'valth': epoch_th}, epoch)		
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

	epoch_acc = 0.0
	epoch_x = 0.0
	epoch_y = 0.0
	epoch_w = 0.0
	epoch_h = 0.0
	epoch_th = 0.0
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			x, label= data
			
			if (GPU):
				x = x.to(device)
				label = label.to(device)

			y = model(x)
			#c_accuracy,c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff = accmetrix(y, label , x.size(0))
			#
			c_accuracy,c_precision,c_recall,x_diff,y_diff,w_diff,h_diff,th_diff = accmetrix(y, label , x.size(0))


			batch_acc = c_accuracy
			epoch_acc += batch_acc * x.size(0)

			epoch_x += x_diff
			epoch_y += y_diff
			epoch_w += w_diff
			epoch_h += h_diff
			epoch_th += th_diff	



		epoch_acc = epoch_acc / tesize
		epoch_x = epoch_x / (tesize/BATCH_SIZE)
		epoch_y = epoch_y / (tesize/BATCH_SIZE)
		epoch_w = epoch_w / (tesize/BATCH_SIZE)
		epoch_h = epoch_h / (tesize/BATCH_SIZE)
		epoch_th = epoch_th / (tesize/BATCH_SIZE)

		print('-' * 20)
		print(' Test Acc: {:.4f}'.format(epoch_acc))
		print('x_diff: {:.4f} y_diff: {:.4f} w_diff: {:.4f} h_diff: {:.4f} th_diff: {:.4f}'.format(x_diff,y_diff,w_diff,h_diff,th_diff))
		model.train(mode=was_training)



if __name__ == '__main__':
	#model = models.resnet18(pretrained = False)
	#fc_features = model.fc.in_features
	#model.fc = nn.Linear(fc_features,24)
	#model.load_state_dict(torch.load('model_para0.05.pkl'))
	model = train()
	#net.load_state_dict(torch.load('model_para.pkl'))
	test(model)

