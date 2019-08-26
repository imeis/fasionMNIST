import torch
import numpy as np
from torchvision import datasets, models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
############################################
# Visualizing part of data set             #
############################################
def vis_data(data_loader, classes):
    """Plots a bunch of data!"""

    # obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(10, 3.5))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        img = np.squeeze(images[idx])
        ax.imshow(img, cmap='gray')
        ax.set_title(classes[labels[idx]])
        
############################################
# helper conv function for residual blocks #
############################################

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


############################################
# Residual Block Class                     #
############################################
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs  
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, batch_norm=True)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, batch_norm=True)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        out_1 = self.dropout(out_1)
        out_2 = x + self.conv_layer2(out_1)
        return out_2


############################################
# Define CNN Architecture                  #
############################################
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# convolutional layer 1 channels in 16 out 
		self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
		# convolutional layer 16 channels in 32 out
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		# convolutional layer 32 channels in 64 out
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

		# outputs 28x28x256 tensor
		self.bn1 = nn.BatchNorm2d(16)
		self.bn2 = nn.BatchNorm2d(32)
		self.bn3 = nn.BatchNorm2d(64)

		# max pooling layer
		self.pool = nn.MaxPool2d(2, 2)

		# dense layer (256 * 7 * 7 -> 128)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		# dense layer (128 -> 10)
		self.fc2 = nn.Linear(128, 10)

		# dropout layer (p=0.4)
		self.dropout = nn.Dropout(0.4)

		# Residual blocks between conv2 and conv3
		res_layers = []
		for layer in range(6):
			res_layers.append(ResidualBlock(32))
		# use sequential to create these layers
		self.res_blocks = nn.Sequential(*res_layers)

	def forward_fc1(self, x):
			for conv in self.h_layers[ : len(self.conv_layers)]:
				x = self.dropout(F.relu(conv(x)))
		
			x = self.pool(x)
		
			x = x.view(x.size(0), -1)
		
			# make the first fc layer if it's the first pass
			if self.first_forward:
				self.first_forward = False
				inp = torch.prod(torch.tensor(x.shape[ 1: ])).item()
				fc1 = nn.Linear(inp, self.fc_layers[0])
				fc1.to(device)
				# insert the layer after the last convolutional layer
				self.h_layers.insert(len(self.conv_layers), fc1)
			fc = self.h_layers[len(self.conv_layers)]
			return F.relu(fc(x))

	def forward(self, x):
		pass
		return x

############################################
# VGG type CNN Struct                      #
############################################
class minivgg(Net):
	def forward(self, x):

		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)

		x = self.conv2(x)        
		x = F.relu(x)
		x = self.pool(x)      
		x = self.bn2(x)

		x = self.conv3(x)
		x = F.relu(x)
		x = self.pool(x)
		x = self.bn3(x)

		x = x.view(-1, 64* 7 * 7)
		x = self.dropout(x)
		x = F.relu(self.fc1(x))

		x = self.dropout(x)
		x = (self.fc2(x))        

		return x    

############################################
#  RESNET type CNN Struct                  #
############################################
class miniresnet(Net):
	def forward(self, x):

		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)

		x = self.conv2(x)        
		x = F.relu(x)
		x = self.pool(x)      
		x = self.bn2(x)

		x = self.res_blocks(x)        

		x = self.conv3(x)
		x = F.relu(x)
		x = self.pool(x)
		x = self.bn3(x)

		x = x.view(-1, 64* 7 * 7)
		x = self.dropout(x)
		x = F.relu(self.fc1(x))

		x = self.dropout(x)
		x = (self.fc2(x))        

		return x    

############################################
# Train                                    #
############################################
def train(n_epochs=50, model=minivgg(), train_loader=None, valid_loader=None,
	optimizer=None, criterion=None, model_name="model_fasionMNIST", train_on_gpu=False):
"""
Training the model.

Arguments:

	n_epochs: Number of epochs in the training
	model: the CNN structure used in the training
	train_loader: the data loader used for the training
	valid_loader: the data loader used for validation of the epoch
	optimizer: the optimizer used in the training
	criterion: used for loss calculations
	model_name: used for saving the model
	train_on_gpu: flag for device

"""

	valid_loss_min = np.Inf # track change in validation loss

	for epoch in range(1, n_epochs+1):

		# keep track of training and validation loss
		train_loss = 0.0
		valid_loss = 0.0

		###################
		# train the model #
		###################
		model.train()

		for data, target in train_loader:
			# move tensors to GPU if CUDA is available
			if train_on_gpu:
				data, target = data.cuda(), target.cuda()
			# clear the gradients of all optimized variables
			optimizer.zero_grad()

			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(data)

			# calculate the batch loss
			loss = criterion(output, target)

			# backward pass: compute gradient of the loss with respect to model parameters
			loss.backward()

			# perform a single optimization step (parameter update)
			optimizer.step()

			# update training loss
			train_loss += loss.item()*data.size(0)

		######################    
		# validate the model #
		######################
		model.eval()

		for data, target in valid_loader:
			# move tensors to GPU if CUDA is available
			if train_on_gpu:
				data, target = data.cuda(), target.cuda()
				# forward pass: compute predicted outputs by passing inputs to the model
			output = model(data)
			
			# calculate the batch loss
			loss = criterion(output, target)
			
			# update average validation loss 
			valid_loss += loss.item()*data.size(0)
		
		# calculate average losses
		train_loss = train_loss/len(train_loader.dataset)
		valid_loss = valid_loss/len(valid_loader.dataset)
		
		# print training/validation statistics 
		print('Epoch: {} Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


		# save model if validation loss has decreased
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,	valid_loss))
			torch.save(model.state_dict(), model_name+'.pt')
			valid_loss_min = valid_loss

############################################
# Test                                     #
############################################
def test_model(model=minivgg(), criterion=None, batch_size=1, test_loader=None, train_on_gpu=False, classes=None):
"""
Testing the model.

Arguments:

	model: the CNN structure used in the testing
	criterion: used for loss calculations
	batch_size: number of batches
	test_loader: the data loader used for the testing
	train_on_gpu: flag for device
	classes: list of the classes

"""

	# track test loss
	test_loss = 0.0
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))

	model.eval()
	counter = 0
	# iterate over test data
	for data, target in test_loader:
		# move tensors to GPU if CUDA is available
		if train_on_gpu:
		  data, target = data.cuda(), target.cuda()
		# forward pass: compute predicted outputs by passing inputs to the model
		output = model(data)
		# calculate the batch loss
		loss = criterion(output, target)
		# update test loss 
		test_loss += loss.item()*data.size(0)
		# convert output probabilities to predicted class
		_, pred = torch.max(output, 1)    
		# compare predictions to true label
		correct_tensor = pred.eq(target.data.view_as(pred))
		correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
		# calculate test accuracy for each object class
		if batch_size == correct.shape[0]:
			for i in range(batch_size):
				 counter =+ 1
				 label = target.data[i]
				 class_correct[label] += correct[i].item()
				 class_total[label] += 1

	# average test loss
	test_loss = test_loss/counter
	print('Test Loss: {:.6f}\n'.format(test_loss))

	for i in range(10):
		if class_total[i] > 0:
			print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
		else:
			print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

	print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))

############################################
# Feature extraction                       #
############################################
def feature_extractor(model, testloader, train_on_gpu, limit=None, category_colors=None):
'''returns the feature layer values and their respective color
Arguments
	model          : the network object
	testloader     : the data loader used for in testing
	limit          : whether to extract all the batches or a limited number
	category_colors: a list of colors, one for each category
	Returns
	feature_values, feature_colors
	''' 
	
	model.eval()

	device = "cpu"
	if train_on_gpu:
		device = "cuda"

	feature_colors = []
	for i, data in enumerate(testloader):
		x, y = data
		x, y = x.to(device), y.to(device)
		if i == 0:
			x_feature = torch.cat((x.view(-1,28*28), ))
		else:
			x_feature = torch.cat((x_feature, x.view(-1,28*28)))
			
		# get the network prediction in order to assign colors 
		# to feature vectors (to which category they belong)
		y_hat = model(x)
		_, predicted = torch.max(y_hat.data, 1)
		feature_colors.append([category_colors[p] for p in predicted])
		# The whole test data is too big for tSNE algorithm to
		# handle all the batches.
		# Only extract the features for the first 50 batches (roughly 1000 images)
		if limit:
			if i == limit:
				 break

	# flatten the feature_colors before return it
	feature_colors = [c for sub_color in feature_colors for c in sub_color]

	model.train()

	return x_feature, feature_colors

############################################
# Plot TSNE                                #
############################################
def plot_tsne(model1, model2, model3, test_loader, train_on_gpu, classes):
"""
Plots the TSNE for three trained models.

Arguments:
	model1, model2, model3: three model objects to be comapred
	testloader     : the data loader used for in testing
	limit          : whether to extract all the batches or a limited number
	category_colors: a list of colors, one for each category
	Returns
	feature_values, feature_colors
"""
	category_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'pink', 'gray', 'brown']
	tsne = TSNE(n_components=2)
	feature1, color1 = feature_extractor(model1, test_loader, train_on_gpu, 50, category_colors)
	feature2, color2 = feature_extractor(model2, test_loader, train_on_gpu, 50, category_colors)
	feature3, color3 = feature_extractor(model3, test_loader, train_on_gpu, 50, category_colors)

	feature1 = feature1.cpu()
	feature2 = feature1.cpu()
	feature3 = feature1.cpu()

	embed1 = tsne.fit_transform(feature1.detach().numpy())
	embed2 = tsne.fit_transform(feature2.detach().numpy())
	embed3 = tsne.fit_transform(feature3.detach().numpy())

	num_classes = len(classes)
	fig, axes = plt.subplots( 1,4,figsize=(18, 6),
					gridspec_kw = {'width_ratios':[4, 4, 4, 1]})

	ax = axes[0]
	ax.set_title('Model1')
	ax.scatter(embed1[:, 0], embed1[:, 1], color=color1, s=2)

	ax = axes[1]
	ax.set_title('Model2')
	ax.scatter(embed2[:, 0], embed2[:, 1], color=color2, s=2)

	ax = axes[2]
	ax.set_title('Model3')
	ax.scatter(embed3[:, 0], embed3[:, 1], color=color3, s=2)

	ax = axes[3]
	ax.axis("off")

	for i in range(num_classes):
		ax.annotate(classes[i], xy=(0, i/10), color=category_colors[i], size=20)

