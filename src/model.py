import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from dataprocess import *
from torchvision import models
import time
import copy
from tensorboardX import SummaryWriter


writer = SummaryWriter(log_dir='scalar')


#dataseg 700 175 batch 25
#580 260 45
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()
EPOCH = 650
BATCH_SIZE = 20
#0.001 0.0003
LR = 0.0035
GPU = True
train_size = 580
val_size = 260
tesize = 45


# id1 train id2 val id3 test
train_data=MyDataset(root=root,datatxt ='id1.txt', transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data=MyDataset(root=root,datatxt ='id2.txt', transform=transforms.ToTensor())
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)


test_data=MyDataset(root=root,datatxt ='id3.txt', transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data, batch_size=45, shuffle=True)

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
def mseloss(y,label,batch):

	newlabel = label
	newlabel = torch.reshape(newlabel,(batch,24))
	newlabel = newlabel.to(torch.float)
	y_n = y.to(torch.float)

	return(mse(y_n,newlabel))
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
		count = len([l for l in iousavetotal if l > 0.5])
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
	

	
	
	model = models.resnet18(pretrained = False)
	fc_features = model.fc.in_features
	model.fc = nn.Linear(fc_features,24)

	best_model_wts = copy.deepcopy(model.state_dict())

	if (GPU):
		model = model.to(device)


	optimizer = optim.Adam(model.parameters(), lr=LR)
	# Decay LR by a factor of 0.1 every 7 epochs
	scheduler = lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.9)

	for epoch in range(EPOCH):
		print('Epoch {}/{}'.format(epoch, EPOCH - 1))
		print('-' * 20)

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
				label = list2tensor(label)
				if (GPU):
					x = x.to(device)
					label = label.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					y = model(x)

					loss = mseloss(y, label , x.size(0))

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
	model = train()
	#net.load_state_dict(torch.load('model_para.pkl'))
	test(model)

