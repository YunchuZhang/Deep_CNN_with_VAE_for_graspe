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
train_size = 40000
val_size = 14000
tesize = 465


# id1 train id2 val id3 test
# 
train_data=MyDataset(root=root,datatxt ='id1.txt',transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data=MyDataset(root=root,datatxt ='id2.txt', transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

test_data=MyDataset(root=root,datatxt ='id3.txt', transform=transforms.Compose([transforms.Resize(640),transforms.ToTensor()]))
test_loader = Data.DataLoader(dataset=test_data, batch_size=465, shuffle=True)

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

	newlabel = torch.reshape(newlabel,(batch,24))
	newlabel = newlabel.to(torch.float)
	y_n = y.to(torch.float)

	return(mse(y_n,newlabel))




	# Reshape input to desired shape
    newlabel = torch.reshape(label,(batch,64,6))
    y_n = torch.reshape(y,(batch,64,6))

    # split the tensor in the order of [prob, x, y, w, h, y, theta]
    cl, xl, yl, wl, hl, thl = torch.split(newlabel, 1, dim=2)
    cp, xp, yp, wp, hp, thp = torch.split(y_n, 1, dim=2)

    # weight different target differently
    lambda_coord = 5
    lambda_obj = 1
    lambda_noobj = 0.3
    mask = cl*lambda_obj+(1-cl)*lambda_noobj
    iou = 
    # linear regression
    lossc = mse(cl*mask*iou, cp*mask*iou)
    lossx = mse(xl*cl*lambda_coord, xp*cl*lambda_coord)
    lossy = mse(yl*cl*lambda_coord, yp*cl*lambda_coord)
    lossw = mse(tensor.sqrt(wl)*cl*lambda_coord, tensor.sqrt(wp)*cl*lambda_coord)
    lossh = mse(tensor.sqrt(hl)*cl*lambda_coord, tensor.sqrt(hp)*cl*lambda_coord)
    lossth = mse(thl*cl*lambda_coord, thp*cl*lambda_coord)
    
    # return joint loss
    loss = lossc+lossx+lossy+lossw+lossh+lossth
	return loss

def acc(y,label,batch):

	iousave = []
	iousavetotal = []
	total = 0.0
	newlabel = label
	y_n = torch.reshape(y,(batch,newlabel.size(1),4,2))


	for i in range(batch):
		for j in range(newlabel.size(1)):
			for k in range(newlabel.size(1)):
				iousave.append(caliou(newlabel[i,j].detach().cpu().numpy(),y_n[i,k].detach().cpu().numpy()))
			iousavetotal.append(max(iousave))
			iousave = []
		count = len([l for l in iousavetotal if l > 0.05])
		total = total + count/newlabel.size(1)*1.0 
		iousavetotal = []

	#print(total)
	return(total)


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
				running_loss += loss.item() * x.size(0)

				running_acc += acc(y, label , x.size(0))

				batch_loss = loss.item() * x.size(0)
				batch_acc = acc(y, label , x.size(0))/BATCH_SIZE

				print('{} Batch_Loss: {:.4f} Batch_Acc: {:.4f}'.format(phase, batch_loss, batch_acc))


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
			label = list2tensor(label)
			if (GPU):
				x = x.to(device)
				label = label.to(device)

			y = model(x)
			running_acc += acc(y, label , x.size(0))
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

