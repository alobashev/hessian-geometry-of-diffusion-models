import os
import time
import pathlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt 

import scipy.special
from sklearn.neighbors import kneighbors_graph

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from src.U2Nets import U2NETP, U2NET
from torch.utils.data import Dataset, DataLoader

DATASETS_PATH = './data/grid_generation/'
os.makedirs('training_progress', exist_ok=True)

# model_name = 'diff'
# Important parameters:
train_test_ratio = 0.8
K_neib = 8
K_bundle = 4
DEVICE_NUMBERS_TO_USE = [0]
BATCH_SIZE = 128
learning_rate = 0.000001
N_epochs = 60

##############
# 
#       0.1 Apply gaussian blur
#
############
alpha = 1 #  image = alpha * image + (1-alpha) * torch.rand(image.shape)*255

model_name = f'diff_clips' #CHANGE HERE

########################################################################################################
#
#    1. Load dataset
#
########################################################################################################
print(f"current path: {pathlib.Path().resolve()}")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
global_datetime = str(now.date().year) + "_" + str(100 + now.date().month)[1:] + "_" + str(now.date().day) + "_" + str(current_time)

states = np.load(DATASETS_PATH+"clip_as_states.npy",)
# states = np.load(DATASETS_PATH+"states.npy",)
labels = np.load(DATASETS_PATH+"labels.npy",)
targets = np.load(DATASETS_PATH+"targets.npy",)

# Assuming your dataset is stored in a variable called `data`
# Example: data = [{"image": img, "alpha": alpha_values[idx], "beta": beta_values[idx]} for idx in range(N)]

########################################################################################################
#
#    2. Split dataset to train and test sets
#
########################################################################################################

print(f"train_test_ratio = {train_test_ratio}")
train_test_cutoff_idx = int(len(labels) * train_test_ratio)
print(f"train_test_cutoff_idx = {train_test_cutoff_idx}")

# Convert numpy arrays to PyTorch tensors
train_X = torch.Tensor(states[:train_test_cutoff_idx])  # Training images
test_X = torch.Tensor(states[train_test_cutoff_idx:])  # Testing images
parameters_train_X = torch.Tensor(labels[:train_test_cutoff_idx])  # Training labels (alpha, beta)
parameters_test_X = torch.Tensor(labels[train_test_cutoff_idx:])  # Testing labels (alpha, beta)
train_Y = torch.Tensor(targets[:train_test_cutoff_idx])  # Training targets (Gaussians)
test_Y = torch.Tensor(targets[train_test_cutoff_idx:])  # Testing targets (Gaussians)

# Print shapes for verification
print(f"train_X.shape = {train_X.shape}")  # Should be (train_size)x128x128x3
print(f"test_X.shape = {test_X.shape}")  # Should be (test_size)x128x128x3
print(f"parameters_train_X.shape = {parameters_train_X.shape}")  # Should be (train_size)x2
print(f"parameters_test_X.shape = {parameters_test_X.shape}")  # Should be (test_size)x2
print(f"train_Y.shape = {train_Y.shape}")  # Should be (train_size)x128x128
print(f"test_Y.shape = {test_Y.shape}")  # Should be (test_size)x128x128
print()

########################################################################################################
#
#    3. Compute list of nearest neighbours for parameters fot train and test
#
########################################################################################################
print(f"K_neib = {K_neib}")

A_train = kneighbors_graph(parameters_train_X, K_neib, mode='connectivity', include_self=True)
neib_list_train = A_train.indices.reshape(-1, K_neib)


A_test = kneighbors_graph(parameters_test_X, K_neib, mode='connectivity', include_self=True)
neib_list_test = A_test.indices.reshape(-1, K_neib)

print(f"neib_list_train.shape = {neib_list_train.shape}")
print(f"neib_list_test.shape = {neib_list_test.shape}")

########################################################################################################
#
#    4. Initialize U2Net model for posterior approximation
#
########################################################################################################

# Input batch of images is constructed by choosing K_bundle images out of K_neib and reshuffling them
#K_bundle = 4
print(f"K_bundle = {K_bundle}")

effective_augmentation_ratio = int(scipy.special.binom(K_neib, K_bundle)*scipy.special.factorial(K_bundle))
print(f"effective_augmentation_ratio = {effective_augmentation_ratio}")

effective_grid_resolution = np.sqrt(len(test_X))/K_neib
print(f"effective_grid_resolution = {effective_grid_resolution:.05}")

N_CHANNELS = train_X.shape[1]
print(f"N_CHANNELS = {N_CHANNELS}")

IN_CHANNELS_U2NET = N_CHANNELS*K_bundle
print(f"IN_CHANNELS_U2NET = {IN_CHANNELS_U2NET}")



print(f"total gpu devices = {torch.cuda.device_count()}")

#DEVICE_NUMBERS_TO_USE = [4,6]
print(f"DEVICE_NUMBERS_TO_USE = {DEVICE_NUMBERS_TO_USE}")

device = torch.device("cuda:"+str(DEVICE_NUMBERS_TO_USE[0]) if (len(DEVICE_NUMBERS_TO_USE) and torch.cuda.device_count() > 0) else "cpu")
print(f"device = {device}")

posterior_net = U2NET(IN_CHANNELS_U2NET, 1).to(device)
print(f"total parameters = {sum([_.numel() for _ in posterior_net.parameters()])/1e6:0.4}")

if (device.type == 'cuda') and (len(DEVICE_NUMBERS_TO_USE) > 1):
	posterior_net = nn.DataParallel(posterior_net, device_ids=DEVICE_NUMBERS_TO_USE, output_device=device, dim=0)
    
 ########################################################################################################
#
#    5. Define dataloaders
#
########################################################################################################

dataset_config = dict()
dataset_config["train_X"] = train_X
dataset_config["test_X"] = test_X
dataset_config["train_Y"] = train_Y
dataset_config["test_Y"] = test_Y
dataset_config["neib_list_train"] = neib_list_train
dataset_config["neib_list_test"] = neib_list_test
dataset_config["K_bundle"] = K_bundle

class MicrostatesDataset(Dataset):
    def __init__(self, dataset_config, train=True):
        if train:
            self.inputs = dataset_config["train_X"]  # Training images
            self.targets = dataset_config["train_Y"]  # Training targets
            self.neib_list = dataset_config["neib_list_train"]  # Nearest neighbors for training
            self.k_bundle = dataset_config["K_bundle"]  # Number of images to bundle
        else:
            self.inputs = dataset_config["test_X"]  # Testing images
            self.targets = dataset_config["test_Y"]  # Testing targets
            self.neib_list = dataset_config["neib_list_test"]  # Nearest neighbors for testing
            self.k_bundle = dataset_config["K_bundle"]  # Number of images to bundle
        
        # Assuming inputs are of shape Nx128x128x3 (images)
        print(self.inputs.shape)
        self.image_size = self.inputs.shape[2]  # Height/width of the image (128)
        self.num_channels = self.inputs.shape[1]  # Number of channels (3)
        print(f'self.image_size{self.image_size}, self.num_channels{self.num_channels}')
    def __getitem__(self, index):
        # Sample K_bundle neighbors from the nearest neighbors list
        sampled_neighbours = np.random.choice(self.neib_list[index], size=self.k_bundle, replace=False)
        
        # Get the images for the sampled neighbors
        image = self.inputs[sampled_neighbours]  # Shape: K_bundle x 3 x 128 x 128 
        # # Permute to channels-first format: K_bundle x 3 x 128 x 128
        # image = image.permute(0, 3, 1, 2)
        # Reshape the images into a single tensor: (K_bundle * num_channels) x 128 x 128
        image = image.reshape(self.k_bundle * self.num_channels, self.image_size, self.image_size)
        
        # Get the targets for the sampled neighbors
        target = self.targets[sampled_neighbours]  # Shape: K_bundle x 128 x 128
        # Average the targets across the sampled neighbors
        target = target.mean(0)  # Shape: 128 x 128
        
        return image, target
    
    def __len__(self):
        return len(self.inputs)

# Create dataset instances
dataset_train = MicrostatesDataset(dataset_config, train=True)
dataset_test = MicrostatesDataset(dataset_config, train=False)

# Create dataloaders
trainloader = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    pin_memory=False
)

testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    pin_memory=False
)

# Verify the shapes of the data
image, target = next(iter(trainloader))
print(f"train: image.shape = {image.shape}")  # Should be (BATCH_SIZE, K_bundle * 3, 128, 128)
print(f"train: target.shape = {target.shape}")  # Should be (BATCH_SIZE, 128, 128)

image, target = next(iter(testloader))
print(f"test: image.shape = {image.shape}")  # Should be (BATCH_SIZE, K_bundle * 3, 128, 128)
print(f"test: target.shape = {target.shape}")  # Should be (BATCH_SIZE, 128, 128)

########################################################################################################
#
#    6. Define loss function
#
########################################################################################################
    
class SoftmaxProbs(nn.Module):
	#Custom Linear layer but mimics a standard linear layer
	def __init__(self):
		super().__init__()
	def forward(self, res):
		return F.softmax(torch.flatten(res, start_dim=-2), dim=-1).view(*res.shape)
    
softmax_probs = SoftmaxProbs()

target_posterior = target.to(device)
# Stochastic label smoothing (optional)
target_posterior = softmax_probs(torch.abs(torch.randn_like(target_posterior))/50+target_posterior)
with torch.no_grad():
    predicted_posterior = softmax_probs(posterior_net(image.to(device)))
    # Negative log likelihodd loss
    loss = -torch.sum(target_posterior*torch.log(predicted_posterior))/BATCH_SIZE

print(f"untrained loss = {loss}")


########################################################################################################
#
#    7. Define optimizer
#
########################################################################################################

#learning_rate = 0.000001
print(f"learning_rate = {learning_rate}")

optimizer_posterior = optim.Adam(posterior_net.parameters(), lr=learning_rate)
losses_train = []
losses_test = []

print(f"len(trainloader) = {len(trainloader)}")
print(f"len(trainloader)*BATCH_SIZE = {len(trainloader)*BATCH_SIZE}")
print(f"len(dataset_train)/BATCH_SIZE = {len(dataset_train)/BATCH_SIZE}")

########################################################################################################
#
#    8. Training loop
#
########################################################################################################

train_loss = []
test_loss = []

# posterior_net.load_state_dict(torch.load('training_progress/checkpoint__model_diff_eta25_start_time_2025_01_27_01_19_01_epoch_100055_current_time_2025_01_27_04_50_35.pth')['posterior_net'])


for epoch in range(N_epochs):
    
    #######################################################################
    #
    #                       Train iterations
    # 
    #######################################################################
    
    train_loss_tmp = []
    for image, target in tqdm(trainloader, position=0, leave=True, desc=f"trainig epoch = {epoch}"):

        image = alpha * image + (1-alpha) * torch.rand(image.shape)*255
        
        optimizer_posterior.zero_grad()
        predicted_posterior = softmax_probs(posterior_net(image.to(device)))
        target_posterior = target.to(device)
        loss = -torch.sum(target_posterior*torch.log(predicted_posterior))/BATCH_SIZE
        loss.backward()
        optimizer_posterior.step()
        
        train_loss_tmp.append(loss.item())
        
    train_loss.append(np.mean(train_loss_tmp))
    
    #######################################################################
    #
    #                       Test iterations
    # 
    #######################################################################
    
    test_loss_tmp = []
    with torch.no_grad():
        for image, target in tqdm(testloader, position=0, leave=True, desc=f"testing epoch = {epoch}"):
            image = alpha * image + (1-alpha) * torch.rand(image.shape)*255
            predicted_posterior = softmax_probs(posterior_net(image.to(device)))
            target_posterior = target.to(device)
            loss = -torch.sum(target_posterior*torch.log(predicted_posterior))/BATCH_SIZE
            
            test_loss_tmp.append(loss.item())
        test_loss.append(np.mean(test_loss_tmp))
        
    if (epoch)%5 == 0 or epoch == N_epochs-1:
        #######################################################################
        #
        #                       Checkpoint saving
        # 
        #######################################################################
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_datetime = str(now.date().year)+"_"+str(100+now.date().month)[1:]+"_"+str(now.date().day)+"_"+str(current_time)
    
        with torch.no_grad():
            image, target = next(iter(testloader))
            predicted_posterior = softmax_probs(posterior_net(image.to(device)))
            predicted_posterior_example = predicted_posterior.cpu().detach()
        
        checkpoint = { 
            'epoch': epoch,
            'posterior_net':posterior_net.state_dict(),
            'optimizer_posterior': optimizer_posterior.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'datetime': datetime,
            'predicted_posterior_example': predicted_posterior_example,
            'global_datetime_of_experiment_start': global_datetime,
        }
        # Sanitize the datetime strings (replace colons with underscores)
        global_datetime = global_datetime.replace(':', '_')
        current_datetime = current_datetime.replace(':', '_')
        checkpoint_path = f'training_progress/checkpoint__model_{model_name}_start_time_{global_datetime}_epoch_{100000+epoch}_current_time_{current_datetime}.pth'
        print(f"Saving checkpoint to: {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
