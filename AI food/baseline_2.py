from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import random
import numpy as np
# import matplotlib.pyplot as plt

#---------------------------------------------------------------

from PIL import Image
    # data augment
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_trans_ImageNet = transforms.Compose([
    # transforms.RandomCrop(size=220,pad_if_needed=True),
    transforms.Resize((224,224)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.05,hue=0),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.90,1.10)),
    transforms.ToTensor(),
    norm,
    Cutout(n_holes=2,length=64)
])
val_trans_ImageNet = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    norm
])
#---------------------------------------------------------------
# dataTrans = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])

# image data path
data_dir = 'ai-food-data/images'
class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

def split_Train_Val_Data(data_dir, ratio, bs=32):
    global train_len
    global val_len
    """ the sum of ratio must equal to 1"""

    dataset = datasets.ImageFolder(data_dir)     # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):   # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        #print(num_sample_train)
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val
        # split into list
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
    
    train_len = len(train_inputs)
    val_len = len(val_inputs)
    print("train_length:%d,val length:%d" %(train_len,val_len))
    
    train_dst = MyDataset(train_inputs, train_labels, train_trans_ImageNet)
    valid_dst = MyDataset(val_inputs, val_labels, val_trans_ImageNet)
    train_dataloader = DataLoader(train_dst,
                                  batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(valid_dst,
                                  batch_size=bs, shuffle=False)
 
    return train_dataloader, val_dataloader

# 定义pytorch的dataloader，数据划分0.9 可以提升一个点左右
data_loader = split_Train_Val_Data(data_dir,(0.9,0.1))

# 为了保证后面和官方的baseline一致，所以可以这么写
dataloders = {x:  data_loader[i] for i,x in enumerate(['train', 'val']) }
dataset_sizes = {'train':train_len, 'val':val_len}
print(dataset_sizes)

# all_image_datasets = datasets.ImageFolder(data_dir, train_trans_Imagenet)

# print(all_image_datasets.class_to_idx)  
# random地划分训练集和验证集  
# trainsize = int(0.9*len(all_image_datasets))
# testsize = len(all_image_datasets) - trainsize
# train_dataset, test_dataset = torch.utils.data.random_split(all_image_datasets,[trainsize,testsize])
# image_datasets = {'train':train_dataset,'val':test_dataset}

    # wrap your data and label into Tensor
#dataloader
# dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
#                                                  batch_size=32,
#                                                  shuffle=True,
#                                                  num_workers=4) for x in ['train', 'val']}

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # use gpu or not
use_gpu = torch.cuda.is_available()

#---------------------------------------------------------------

def train_model(model, lossfunc, optimizer, scheduler, num_epochs=10):
    start_time = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                try:
                    outputs = model(inputs)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("warning: out of memory")
                        if hasattr(torch.cuda,'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                _, preds = torch.max(outputs.data, 1)
                loss = lossfunc(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                                        
                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase=='val':
                val_acc.append(epoch_acc)
            else:
                train_acc.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step(val_acc[-1])
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc, val_acc

#---------------------------------------------------------------

# get model and replace the original fc layer with your fc layer
model_ft = models.resnext50_32x4d(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

if use_gpu:
    model_ft = model_ft.cuda()

    # define loss function
lossfunc = nn.CrossEntropyLoss()

    # setting optimizer and   #！！trainable parameters
#   params = model_ft.parameters()
# list(model_ft.fc.parameters())+list(model_ft.layer4.parameters())
#params = list(model_ft.fc.parameters())+list( model_ft.parameters())
params = list(model_ft.parameters())
lr = 0.001
optimizer_ft = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
print('learning rate {}'.format(lr))
    # Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', patience=2, verbose=True)
model_ft, train_acc, val_acc = train_model(model=model_ft,
                           lossfunc=lossfunc,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=20)

#---------------------------------------------------------------

model_name = 'model_l.pth'
print("save as", model_name)
torch.save(model_ft.state_dict(), model_name)

#---------------------------------------------------------------
                          
    # plot
# plt.plot(train_acc, label="train")
# plt.plot(val_acc, label="val")
# plt.legend()
# plt.plot()
# print(train_acc)
# print(val_acc)

