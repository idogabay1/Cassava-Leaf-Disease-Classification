# -*- coding: utf-8 -*-
"""Cassava Leaf Disease Classification on balanced test-set
"""

###################### READ ME #######################
#This project solve the classification of Cassava leaf diseases from biased train-set to balanced test-set.
#The dataset is based on the “Cassava Leaf Disease Classification” dataset from Kaggle.
#You can find it here : https://www.kaggle.com/c/cassava-leaf-disease-classification
####NOTE that for some application we extended this dataset. If you wish to use the extended dataset you can do so using the script we provide in this documentation.
#We used Google Colab for running our code so all of it designed for it.
#Enjoy 😊
#####################################################

### This line solves compatibility issues with processing the linear layers. It take some times but necessary.

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

### Installing Optuna. This is not necessary if you do not intend to use Optuna for building a new architecture and only use the provided architectures.

pip install optuna

# Commented out IPython magic to ensure Python compatibility.
### Python imports
# imports for the practice (you can add more if you need)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch.utils.data.sampler import SubsetRandomSampler as sub_rand_sampler
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import seaborn as sns
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
import optuna
from random import seed
from random import random

# %matplotlib notebook
# %matplotlib inline
### random seed for later.
seed = 70
np.random.seed(seed)
torch.manual_seed(seed)

### Generating the extended dataset.
### We did it one time locally to our computer and with this data generated the zip file.
### To use this script make a copy of the data because it’s going to delete the renamed files.

pic_name = []
pic_label = []
original_pic_name = []
f1 = open("./dataset-train-not-class3.csv",'r') ###csv without class 3 samples.
reader = csv.reader(f1)
for row in reader:
    original_pic_name.append(row[0])
    pic_name.append(row[0]+'_copy1.jpg')
    pic_label.append(row[1])

f2 = open("./non3_copy_1.csv",'r+')
writer = csv.writer(f2)
for i in range(len(pic_name)):
    writer.writerow([pic_name[i],pic_label[i]])

for count, filename in enumerate(os.listdir("800_600_full")):
    
    if filename in original_pic_name:
        src ="dataset/"+filename
        dst ="dataset_extension1/"+filename+'_copy1.jpg'
        os.rename(src, dst)

f1.close
f2.close

### We executed this script 3 times, each time with different extension number (change 1,2 and 3 in all the places).

### Configure the computation device 

# check if there is a GPU available
print(torch.cuda.is_available())
# check what is the current available device
if torch.cuda.is_available():
    print("current device: ", torch.cuda.current_device())
# automatically choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use gpu 0 if it is available, o.w. use the cpu
print("device: ", device)

### Mount your Google Drive in Colab

from google.colab import drive
drive.mount('/content/drive')

### Imports the dataset from  Google Drive. We found it the most efficient way to transfer the data to Google Colab many times.
###To do so, save the dataset as a zip file in the name “dataset.zip” in Google Drive.

! mkdir "dataset" #create a new folder
!cp "/content/drive/MyDrive/YOUR_PATH/dataset.zip" "/content/dataset" #copy the zip file
! unzip '/content/dataset/dataset.zip' -d "/content/dataset" #unzip

###Note that we included in our zip file the extended dataset. This is not necessary

image_path = "/content/dataset" #dataset images path


### This comment contain several csv paths including the extensions.
###We do so to make several datasets for implying different augmentations on each extension.
###Use it only if you extended the data.
"""
train_csv_without_class3 = "/content/drive/MyDrive/CSV_PATH/dataset-train-not-class3.csv"
train_csv_without_class3_copy1 = "/content/drive/MyDrive/CSV_PATH/dataset_train_non3_extension_1.csv"
train_csv_without_class3_copy2 = "/content/drive/MyDrive/CSV_PATH/dataset_train_non3_extension_2.csv"
train_csv_without_class3_copy3 = "/content/drive/MyDrive//dataset_train_non3_extension_3.csv"
train_csv_only_class3 = "/content/drive/MyDrive/deep_final_project/csv/dataset-train-only-class3.csv"
test_csv_path = "/content/drive/MyDrive/deep_final_project/csv/dataset-test.csv"
"""

### Note that use use imbalanced train-set and balanced test-set.
train_csv = "/content/drive/MyDrive/CSV_PATH/dataset-train.csv"
test_csv_path = "/content/drive/MyDrive/CSV_PATH/dataset-mini-test.csv"

### Show the begining of the CSV 
csv_file = pd.read_csv(train_csv)
csv_file.head()

### Dataset class

class Cassava_Dataset(Dataset):
    
    def __init__(self, image_path,csv_file, Transform=None):
        self.image_path = image_path
        self.csv = pd.read_csv(csv_file) 
        self.transform = Transform
        
    def __len__(self): 
      return len(self.csv)
    
    def __getitem__(self, idx): 
      if torch.is_tensor(idx):
        idx = idx.tolist()
       
      image = Image.open(os.path.join(self.image_path, self.csv.iloc[idx,0]))
      label = self.csv.iloc[idx,1]
        #if self.image_path == '../input/cassava-leaf-disease-classification/train_images':
      return self.transform(image), label

### augmentation setup.
### You can choose the transform you want. The basic one only normalize the data and turn it in to tensor.

basic_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])#transforms.Resize((200,150)),

transform1 = transforms.Compose([transforms.Resize((224,224)),transforms.RandomRotation((7,15)),
                                 torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform1_point_5 = transforms.Compose([transforms.Resize((224,224)),transforms.RandomRotation((7,15)),
                                 torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transform2 = transforms.Compose([transforms.Resize((224,224)),torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=1),
                                 transforms.RandomRotation((-15,-7)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transform3 = transforms.Compose([transforms.Resize((224,224)),torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=1),
                                 transforms.RandomRotation((7,15)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

transform4 = transforms.Compose([transforms.Resize((224,224)),torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=1),
                                 transforms.RandomRotation((12,17)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

crop_transform = transforms.Compose([transforms.Resize((224,224)),torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.5),
                                     torchvision.transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                     torchvision.transforms.RandomCrop((180,180)),#, padding=48, fill=0, padding_mode='constant')])
                                     torchvision.transforms.Pad(22, fill=0, padding_mode='constant')])

transform_resnet_basic = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

class_3_transform = basic_transform
train_transform = basic_transform
test_transform = basic_transform


### Note that the commented part is for the extended data.
"""
train_without_class3_dataset1 = Cassava_Dataset(image_path,train_csv_without_class3,all_trandform)
train_only_3_class_dataset = Cassava_Dataset(image_path,train_csv_only_class3,all_trandform)

train_without_class3_dataset2 = Cassava_Dataset(image_path,train_csv_without_class3_copy1,all_trandform)
train_without_class3_dataset3 = Cassava_Dataset(image_path,train_csv_without_class3_copy2,all_trandform)
train_without_class3_dataset4 = Cassava_Dataset(image_path,train_csv_without_class3_copy3,all_trandform)


train_dataset = torch.utils.data.ConcatDataset((train_without_class3_dataset4,train_without_class3_dataset1,
                                                train_only_3_class_dataset, train_without_class3_dataset2,
                                                train_without_class3_dataset3))
"""

##for optuna
train_dataset = Cassava_Dataset(image_path,train_csv,train_transform)



test_dataset = Cassava_Dataset(image_path,test_csv_path,test_transform)

batch_size = 128

### split the test to validation and test
val_len = int(len(test_dataset)*0.5)
test_len = len(test_dataset) - val_len
test_set, valid_set = random_split(test_dataset, [test_len, val_len])


train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

############## Network Selection ##################
### In this project we tried several architectures.
### in the following sections YOU NEED TO CHOOSE ONLY ONE and change the model name in the right place afterward.

### fork of cassva model. copied from : "https://www.kaggle.com/charlesrongione/fork-of-cassava"

class Fork_of_cassava(nn.Module):
  def __init__(self):
    super().__init__()
        
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,padding=1)
    self.conv2 = nn.Conv2d(in_channels=12,out_channels=24,kernel_size=3,padding=1)
    self.fc1 = nn.Linear(180000, 512)
    self.fc2 = nn.Linear(512, 5)       
  
  
  def forward(self,x):
    x = nn.functional.relu(self.conv1(x))
    x = torch.flatten(nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)),2),1)
    x = self.fc1(x)
    x = nn.functional.relu(x)
    x = self.fc2(x)
    return output

### Our attempt to improve cassava model using Batch norm, dropout, pooling, and STN layer.
### You can find more information on STN here : " https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html”

class Fork_of_cassava_improved_with_stn(nn.Module):
  def __init__(self):
    super(Fork_of_cassava_improved_with_stn, self).__init__()
    
    ##STN block
    self.localization = nn.Sequential(nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),nn.ReLU(True),nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),nn.ReLU(True))   
    # Regressor for the 3 * 2 affine matrix
    self.fc_loc = nn.Sequential(nn.Linear ( 2001920, 32),nn.ReLU(True),
        nn.Linear(32, 3 * 2))

    self.fc_loc[2].weight.data.zero_()
    self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))         
    ##end of stn block


    self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)
    self.baNorm1 = nn.BatchNorm2d(32)
    self.max_pool1 = nn.MaxPool2d((2,2))
    self.drop1 = nn.Dropout(0.1)
    
    self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
    self.baNorm2 = nn.BatchNorm2d(64)
    self.max_pool2 = nn.MaxPool2d((2,2))
    self.drop2 = nn.Dropout(0.1)

    self.fc1 = nn.Linear(110592, 216)
    self.BaNorm5=nn.BatchNorm1d(216)
    self.drop5=nn.Dropout(0.1)
    
    self.fc2 = nn.Linear(216, 5)  

  def stn(self, x):
    xs = self.localization(x)
    xs = xs.view(-1, 2001920)
    theta = self.fc_loc(xs)
    theta = theta.view(-1, 2, 3)
    print('bi')
    grid = F.affine_grid(theta, x.size())
    print('sha')
    x = F.grid_sample(x, grid)
    print('va')
    return x
  
  def forward(self,x):
    x = self.stn(x)
    x = self.max_pool1(nn.functional.relu(self.baNorm1(self.conv1(x))))
    x = self.drop1(x)

    x = self.max_pool2(nn.functional.relu(self.baNorm2(self.conv2(x))))
    x = self.drop2(x)
    x = torch.flatten(x,1)
    x = self.drop5(torch.relu(self.BaNorm5(self.fc1(x))))

    x = self.fc2(x)
    return x

### Optuna model suggestion number 2

class optuna_res_2(nn.Module):
  def __init__(self):
    """CNN Builder."""
    super(optuna_res_2, self).__init__()
    #the network
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=92,kernel_size=3,padding=1)
    self.baNorm1 = nn.BatchNorm2d(92)
    self.drop1 = nn.Dropout(0)

    self.conv2 = nn.Conv2d(in_channels=92,out_channels=76,kernel_size=3,padding=1)
    self.baNorm2 = nn.BatchNorm2d(76)
    self.drop2 = nn.Dropout(0)

    self.conv3 = nn.Conv2d(in_channels=76,out_channels=78,kernel_size=3,padding=1)
    self.baNorm3 = nn.BatchNorm2d(78)
    self.drop3 = nn.Dropout(0)
    
    self.conv4 = nn.Conv2d(in_channels=78,out_channels=44,kernel_size=3,padding=1)
    self.baNorm4 = nn.BatchNorm2d(44)
    self.drop4 = nn.Dropout(0)

    self.conv5 = nn.Conv2d(in_channels=44,out_channels=71,kernel_size=3,padding=1)
    self.baNorm5 = nn.BatchNorm2d(71)
    self.drop5 = nn.Dropout(0)

    self.conv6 = nn.Conv2d(in_channels=71,out_channels=120,kernel_size=3,padding=1)
    self.baNorm6 = nn.BatchNorm2d(120)
    self.drop6 = nn.Dropout(0)

    self.conv7 = nn.Conv2d(in_channels=120,out_channels=27,kernel_size=3,padding=1)
    self.baNorm7 = nn.BatchNorm2d(27)
    self.drop7 = nn.Dropout(0)

    self.conv8 = nn.Conv2d(in_channels=27,out_channels=104,kernel_size=3,padding=1)
    self.baNorm8 = nn.BatchNorm2d(104)
    self.drop8 = nn.Dropout(0)

    self.conv9 = nn.Conv2d(in_channels=104,out_channels=66,kernel_size=3,padding=1)
    self.baNorm9 = nn.BatchNorm2d(66)
    self.drop9 = nn.Dropout(0)
    
    self.conv10 = nn.Conv2d(in_channels=66,out_channels=57,kernel_size=3,padding=1)
    self.baNorm10 = nn.BatchNorm2d(57)
    self.drop10 = nn.Dropout(0)

    self.conv11 = nn.Conv2d(in_channels=57,out_channels=32,kernel_size=3,padding=1)
    self.baNorm11 = nn.BatchNorm2d(32)
    self.drop11 = nn.Dropout(0)

    self.conv12 = nn.Conv2d(in_channels=32,out_channels=57,kernel_size=3,padding=1)
    self.baNorm12 = nn.BatchNorm2d(57)
    self.drop12 = nn.Dropout(0)

    self.conv13 = nn.Conv2d(in_channels=57,out_channels=118,kernel_size=3,padding=1)
    self.baNorm13 = nn.BatchNorm2d(118)
    self.drop13 = nn.Dropout(0)

    self.conv14 = nn.Conv2d(in_channels=118,out_channels=48,kernel_size=3,padding=1)
    self.baNorm14 = nn.BatchNorm2d(48)
    self.drop14 = nn.Dropout(0)
    

    self.conv15 = nn.Conv2d(in_channels=48,out_channels=27,kernel_size=3,padding=1)
    self.baNorm15 = nn.BatchNorm2d(27)
    self.drop15 = nn.Dropout(0)

    self.fc1 = nn.Linear(1354752, 5)
    

  def forward(self,x):
    x = self.drop1(nn.functional.relu(self.baNorm1(self.conv1(x))))
    x = self.drop2(nn.functional.relu(self.baNorm2(self.conv2(x))))
    x = self.drop3(nn.functional.relu(self.baNorm3(self.conv3(x))))
    x = self.drop4(nn.functional.relu(self.baNorm4(self.conv4(x))))
    x = self.drop5(nn.functional.relu(self.baNorm5(self.conv5(x))))
    x = self.drop6(nn.functional.relu(self.baNorm6(self.conv6(x))))
    x = self.drop7(nn.functional.relu(self.baNorm7(self.conv7(x))))
    x = self.drop8(nn.functional.relu(self.baNorm8(self.conv8(x))))
    x = self.drop9(nn.functional.relu(self.baNorm9(self.conv9(x))))
    x = self.drop10(nn.functional.relu(self.baNorm10(self.conv10(x))))
    x = self.drop11(nn.functional.relu(self.baNorm11(self.conv11(x))))
    x = self.drop12(nn.functional.relu(self.baNorm12(self.conv12(x))))
    x = self.drop13(nn.functional.relu(self.baNorm13(self.conv13(x))))
    x = self.drop14(nn.functional.relu(self.baNorm14(self.conv14(x))))
    x = self.drop15(nn.functional.relu(self.baNorm15(self.conv15(x))))
    
    x = torch.flatten(x,1)
    x = self.fc1(x)
    
    return x

### Optuna model suggestion number 2 with skip connections and pooling layers

class optuna_res_2_skip_pool(nn.Module):
  def __init__(self):
    """CNN Builder."""
    super(optuna_res_2_skip_pool, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3,out_channels=50,kernel_size=3,padding=1)
    self.baNorm1 = nn.BatchNorm2d(50)
    self.drop1 = nn.Dropout(0)

    self.conv2 = nn.Conv2d(in_channels=50,out_channels=76,kernel_size=3,padding=1)
    self.baNorm2 = nn.BatchNorm2d(76)
    self.drop2 = nn.Dropout(0)

    self.conv3 = nn.Conv2d(in_channels=76,out_channels=50,kernel_size=3,padding=1)
    self.baNorm3 = nn.BatchNorm2d(50)
    self.drop3 = nn.Dropout(0)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = nn.Conv2d(in_channels=50,out_channels=44,kernel_size=3,padding=1)
    self.baNorm4 = nn.BatchNorm2d(44)
    self.drop4 = nn.Dropout(0)

    self.conv5 = nn.Conv2d(in_channels=44,out_channels=50,kernel_size=3,padding=1)
    self.baNorm5 = nn.BatchNorm2d(50)
    self.drop5 = nn.Dropout(0)
    self.pool5 = nn.MaxPool2d(kernel_size=2)

    self.conv6 = nn.Conv2d(in_channels=50,out_channels=120,kernel_size=3,padding=1)
    self.baNorm6 = nn.BatchNorm2d(120)
    self.drop6 = nn.Dropout(0)

    self.conv7 = nn.Conv2d(in_channels=120,out_channels=50,kernel_size=3,padding=1)
    self.baNorm7 = nn.BatchNorm2d(50)
    self.drop7 = nn.Dropout(0)
    self.pool7 = nn.MaxPool2d(kernel_size=2)

    self.conv8 = nn.Conv2d(in_channels=50,out_channels=104,kernel_size=3,padding=1)
    self.baNorm8 = nn.BatchNorm2d(104)
    self.drop8 = nn.Dropout(0)

    self.conv9 = nn.Conv2d(in_channels=104,out_channels=50,kernel_size=3,padding=1)
    self.baNorm9 = nn.BatchNorm2d(50)
    self.drop9 = nn.Dropout(0)
    self.pool9 = nn.MaxPool2d(kernel_size=2)

    self.conv10 = nn.Conv2d(in_channels=50,out_channels=57,kernel_size=3,padding=1)
    self.baNorm10 = nn.BatchNorm2d(57)
    self.drop10 = nn.Dropout(0)

    self.conv11 = nn.Conv2d(in_channels=57,out_channels=50,kernel_size=3,padding=1)
    self.baNorm11 = nn.BatchNorm2d(50)
    self.drop11 = nn.Dropout(0)
    self.pool11 = nn.MaxPool2d(kernel_size=2)

    self.conv12 = nn.Conv2d(in_channels=50,out_channels=57,kernel_size=3,padding=1)
    self.baNorm12 = nn.BatchNorm2d(57)
    self.drop12 = nn.Dropout(0)

    self.conv13 = nn.Conv2d(in_channels=57,out_channels=50,kernel_size=3,padding=1)
    self.baNorm13 = nn.BatchNorm2d(50)
    self.drop13 = nn.Dropout(0)
    self.pool13 = nn.MaxPool2d(kernel_size=2)

    self.conv14 = nn.Conv2d(in_channels=50,out_channels=48,kernel_size=3,padding=1)
    self.baNorm14 = nn.BatchNorm2d(48)
    self.drop14 = nn.Dropout(0)
    

    self.conv15 = nn.Conv2d(in_channels=48,out_channels=50,kernel_size=3,padding=1)
    self.baNorm15 = nn.BatchNorm2d(50)
    self.drop15 = nn.Dropout(0)
    self.pool15 = nn.MaxPool2d(kernel_size=2)

    self.fc1 = nn.Linear(50, 5)
    

  def forward(self,x):
    x = self.drop1(nn.functional.relu(self.baNorm1(self.conv1(x))))
    temp = x
    x = self.drop2(nn.functional.relu(self.baNorm2(self.conv2(x))))
    x = self.baNorm3(self.conv3(x))
    x+=temp
    x = self.drop3(self.pool3(nn.functional.relu(x)))

    temp = x
    x = self.drop4(nn.functional.relu(self.baNorm4(self.conv4(x))))
    x = self.baNorm5(self.conv5(x))
    x+=temp
    x = self.drop5(self.pool5(nn.functional.relu(x)))

    temp = x
    x = self.drop6(nn.functional.relu(self.baNorm6(self.conv6(x))))
    x = self.baNorm7(self.conv7(x))
    x+=temp
    x = self.drop7(self.pool7(nn.functional.relu(x)))

    temp = x
    x = self.drop8(nn.functional.relu(self.baNorm8(self.conv8(x))))
    x = self.baNorm9(self.conv9(x))
    x+=temp
    x = self.drop9(self.pool9(nn.functional.relu(x)))

    temp = x    
    x = self.drop10(nn.functional.relu(self.baNorm10(self.conv10(x))))
    x = self.baNorm11(self.conv11(x))
    x+=temp
    x = self.drop11(self.pool11(nn.functional.relu(x)))

    temp = x   
    x = self.drop12(nn.functional.relu(self.baNorm12(self.conv12(x))))
    x = self.baNorm13(self.conv13(x))
    x+=temp
    x = self.drop13(self.pool13(nn.functional.relu(x)))
    
    temp = x   
    x = self.drop14(nn.functional.relu(self.baNorm14(self.conv14(x))))
    x = self.baNorm15(self.conv15(x))
    x+=temp
    x = self.drop15(self.pool15(nn.functional.relu(x)))

    x = torch.flatten(x,1)
    x = self.fc1(x)
    
    return x

### Optuna model suggestion number 2 with stchastic depth and pooling layers.

class optuna_res_2_skip_pool_st_depth(nn.Module):
  def __init__(self):
    """CNN Builder."""
    super(optuna_res_2_skip_pool_st_depth, self).__init__()
    #the network
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=50,kernel_size=3,padding=1)
    self.baNorm1 = nn.BatchNorm2d(50)
    self.drop1 = nn.Dropout(0)

    self.conv2 = nn.Conv2d(in_channels=50,out_channels=76,kernel_size=3,padding=1)
    self.baNorm2 = nn.BatchNorm2d(76)
    self.drop2 = nn.Dropout(0)

    self.conv3 = nn.Conv2d(in_channels=76,out_channels=50,kernel_size=3,padding=1)
    self.baNorm3 = nn.BatchNorm2d(50)
    self.drop3 = nn.Dropout(0)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = nn.Conv2d(in_channels=50,out_channels=44,kernel_size=3,padding=1)
    self.baNorm4 = nn.BatchNorm2d(44)
    self.drop4 = nn.Dropout(0)

    self.conv5 = nn.Conv2d(in_channels=44,out_channels=50,kernel_size=3,padding=1)
    self.baNorm5 = nn.BatchNorm2d(50)
    self.drop5 = nn.Dropout(0)
    self.pool5 = nn.MaxPool2d(kernel_size=2)

    self.conv6 = nn.Conv2d(in_channels=50,out_channels=120,kernel_size=3,padding=1)
    self.baNorm6 = nn.BatchNorm2d(120)
    self.drop6 = nn.Dropout(0)

    self.conv7 = nn.Conv2d(in_channels=120,out_channels=50,kernel_size=3,padding=1)
    self.baNorm7 = nn.BatchNorm2d(50)
    self.drop7 = nn.Dropout(0)
    self.pool7 = nn.MaxPool2d(kernel_size=2)

    self.conv8 = nn.Conv2d(in_channels=50,out_channels=104,kernel_size=3,padding=1)
    self.baNorm8 = nn.BatchNorm2d(104)
    self.drop8 = nn.Dropout(0)

    self.conv9 = nn.Conv2d(in_channels=104,out_channels=50,kernel_size=3,padding=1)
    self.baNorm9 = nn.BatchNorm2d(50)
    self.drop9 = nn.Dropout(0)
    self.pool9 = nn.MaxPool2d(kernel_size=2)

    self.conv10 = nn.Conv2d(in_channels=50,out_channels=57,kernel_size=3,padding=1)
    self.baNorm10 = nn.BatchNorm2d(57)
    self.drop10 = nn.Dropout(0)

    self.conv11 = nn.Conv2d(in_channels=57,out_channels=50,kernel_size=3,padding=1)
    self.baNorm11 = nn.BatchNorm2d(50)
    self.drop11 = nn.Dropout(0)
    self.pool11 = nn.MaxPool2d(kernel_size=2)

    self.conv12 = nn.Conv2d(in_channels=50,out_channels=57,kernel_size=3,padding=1)
    self.baNorm12 = nn.BatchNorm2d(57)
    self.drop12 = nn.Dropout(0)

    self.conv13 = nn.Conv2d(in_channels=57,out_channels=50,kernel_size=3,padding=1)
    self.baNorm13 = nn.BatchNorm2d(50)
    self.drop13 = nn.Dropout(0)
    self.pool13 = nn.MaxPool2d(kernel_size=2)

    self.conv14 = nn.Conv2d(in_channels=50,out_channels=48,kernel_size=3,padding=1)
    self.baNorm14 = nn.BatchNorm2d(48)
    self.drop14 = nn.Dropout(0)
    

    self.conv15 = nn.Conv2d(in_channels=48,out_channels=50,kernel_size=3,padding=1)
    self.baNorm15 = nn.BatchNorm2d(50)
    self.drop15 = nn.Dropout(0)
    self.pool15 = nn.MaxPool2d(kernel_size=2)

    self.fc1 = nn.Linear(50, 5)
    

  def forward(self,x):

    
    x = (self.drop1(nn.functional.relu(self.baNorm1(self.conv1(x))))).to(device)
    temp = x.to(device)
    rand_pass = random()
    if rand_pass > 0.1:
      x = self.drop2(nn.functional.relu(self.baNorm2(self.conv2(x))))
      x = self.baNorm3(self.conv3(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop3(self.pool3(nn.functional.relu(x)))

    temp = x.to(device)
    rand_pass = random()
    if rand_pass > 0.2:
      x = self.drop4(nn.functional.relu(self.baNorm4(self.conv4(x))))
      x = self.baNorm5(self.conv5(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop5(self.pool5(nn.functional.relu(x)))

    temp = x.to(device)
    if rand_pass > 0.3:
      x = self.drop6(nn.functional.relu(self.baNorm6(self.conv6(x))))
      x = self.baNorm7(self.conv7(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop7(self.pool7(nn.functional.relu(x)))

    temp = x
    if rand_pass > 0.4:
      x = self.drop8(nn.functional.relu(self.baNorm8(self.conv8(x))))
      x = self.baNorm9(self.conv9(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop9(self.pool9(nn.functional.relu(x)))

    temp = x    
    if rand_pass > 0.5:
      x = self.drop10(nn.functional.relu(self.baNorm10(self.conv10(x))))
      x = self.baNorm11(self.conv11(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop11(self.pool11(nn.functional.relu(x)))

    temp = x 
    if rand_pass > 0.6:  
      x = self.drop12(nn.functional.relu(self.baNorm12(self.conv12(x))))
      x = self.baNorm13(self.conv13(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop13(self.pool13(nn.functional.relu(x)))
    
    temp = x  
    if rand_pass > 0.7:   
      x = self.drop14(nn.functional.relu(self.baNorm14(self.conv14(x))))
      x = self.baNorm15(self.conv15(x))
    else:
      x = (torch.zeros(x.shape)).to(device)
    x+=temp.to(device)
    x = self.drop15(self.pool15(nn.functional.relu(x)))  

    #print(x.shape)
    x = torch.flatten(x,1)
    x = self.fc1(x)
    
    return x

### For applying transfer learning using Resnet-18 use this section

def set_parameter_requires_grad(model, feature_extracting=False):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False
    else:
      for param in model.parameters():
        param.requires_grad = True

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
 # Initialize these variables which will be set in this if statement. Each of these
 # variables is model specific.
  model_ft = None
  input_size = 0 # image size, e.g. (3, 224, 224)
  if model_name == "resnet":
    """ Resnet18"""
    model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5) # replace the last FC layer
    input_size = 224
  else:
    raise NotImplementedError
  return model_ft, input_size

############### END OF MODELS ####################

# function to calcualte accuracy of the model
def calculate_accuracy(my_model, dataloader, device):
    my_model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([5,5], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = my_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1 

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

######## Model setup sections ##########
###choose only one

##### if you choose resnet-18 run only this section

lr = 0.0005
epochs = 30

#model configuration 
model, input = initialize_model(model_name="resnet", num_classes=5, feature_extract=False, use_pretrained=True)
model = model.to(device)

#weighted classes corresponding to class percentage from total train-set
weights = torch.tensor([12.105,6.011,5.514,1,5.106]).to(device)
#loss function
criterion = nn.CrossEntropyLoss()#weight = weights) 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)#,weight_decay=0.000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.5,patience=2,verbose = True,threshold_mode='abs',threshold=0.3)

###if you didn’t choose Resnet18 run only this section. 
###NOTE : Change the name of the model to the one you chose


lr = 0.001
epochs = 30
#seed(1)
#weighted classes corresponding to class percentage from total train-set
weights = torch.tensor([12.105,6.011,5.514,1,5.106]).to(device)
# loss criterion
criterion = nn.CrossEntropyLoss()#weight = weights)

### change model name here###
model = optuna_res_2_skip_pool_st_depth().to(device)

# optimizer - SGD, Adam, RMSProp...
optimizer = torch.optim.SGD(model.parameters(), lr=lr)#,weight_decay=0.000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.2,patience=2,verbose = True,threshold_mode='abs',threshold=0.3)

######## end of model setup sections ##########

# training loop

loss_memory = []
epoch_accucary_max=0

### enter the path you desire the best model to be saved in.
save_path = '/content/drive/MyDrive/YOUR_MODEL_SAVING_PATH'
save_path= os.path.join(save_path,('model_weights.pth'))

for epoch in range(1, epochs + 1):
    model.train()  # put in training mode
    running_loss = 0.0
    for i,data in enumerate(train_loader):
      if(i%50 == 0):
        print('epoch :',epoch, 'batch :',i)
      
      # get the inputs
      inputs, labels = data
      
      # send them to device
      inputs = inputs.to(device)
      labels = labels.to(device)

      # forward + backward + optimize
      outputs = model(inputs)  # forward pass
      loss = criterion(outputs, labels)  # calculate the loss
      optimizer.zero_grad()  # zero the parameter gradients
      loss.backward()  # backpropagation
      optimizer.step()  # update parameters

      running_loss += loss.data.item()

    # Normalizing the loss by the total number of train batches
    running_loss /= len(train_loader)

    valid_accuracy, _ = calculate_accuracy(model, valid_loader, device)
    train_accuracy, _ = calculate_accuracy(model, train_loader, device)
    loss_memory.append(running_loss)
    
    ###the following comment let you change the LR every X epochs. you can use it instead of the scheduler.
    """
    if ((epoch+1)%5 == 0):
      lr = lr*0.5
      optimizer.param_groups[0]['lr'] = lr
    """
    print('epoch :',epoch,"|loss :", running_loss,"|train_acc:",train_accuracy,"|val_acc :",valid_accuracy,"lr :",optimizer.param_groups[0]['lr'])
    scheduler.step(valid_accuracy)

    ### saving best model based on validation accuracy
    if epoch_accucary_max <= valid_accuracy:
      epoch_accucary_max = valid_accuracy
      state = {
        'net': model.state_dict(),
        'epoch': epoch,
      }
      print("valid accuracy improved to : ",epoch_accucary_max)
      torch.save(state,save_path)

    
print('==> Finished Training ...')

### model evaluation 

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(5,5))
    classes = ('0', '1', '2', '3', '4')
    #plt.yticks(range(5), classes)
    heatmap = sns.heatmap(confusion_matrix, annot=True, cmap='cubehelix', yticklabels=classes, xticklabels=classes)
    heatmap.set_title('Confusion Matrix', fontdict={'fontsize':18}, pad=16)
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.show()

# load model, calculate accuracy and confusion matrix

model = model.to(device)
state = torch.load(save_path, map_location=device)
model.load_state_dict(state['net'])

test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
print("test accuracy: {:.3f}%".format(test_accuracy))

# plot confusion matrix
plot_confusion_matrix(confusion_matrix)

########### OPTUNA #############
###the following sections are for operating Optuna only.
### you can find more imformation about Optuna here : "https://optuna.org/"

def define_model(trial):
  # We optimize the number of layers, hidden units and dropout ratio in each layer.
  n_layers = trial.suggest_int("n_layers", 7, 20) # number of layers will be between 1 and 3
  first_conv_maps = trial.suggest_int("first_conv_maps",16,256)
  layers = []
  #in_channels = 224 * 224
  layers.append(nn.Conv2d(in_channels=3,out_channels=first_conv_maps,kernel_size=3,padding = 1))
  layers.append(nn.BatchNorm2d(first_conv_maps))
  layers.append(nn.ReLU())
  drop_layer1 = trial.suggest_float("drop_layer1",0.1,0.5)
  layers.append(nn.Dropout(drop_layer1))
  in_channels = first_conv_maps
  for i in range(n_layers):
    out_channels = trial.suggest_int("n_units_l{}".format(i+2), 16, 128) # number of units will be between 4 and 128
    layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding = 1))   #nn.Linear(in_features, out_features))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    p = trial.suggest_float("dropout_l{}".format(i+2), 0.1, 0.5) # dropout rate will be between 0.2 and 0.5
    layers.append(nn.Dropout(p))
    in_channels = out_channels

  last_conv_channels = trial.suggest_int("last_conv_maps",4,64)
  layers.append(nn.Conv2d(in_channels=in_channels,out_channels=last_conv_channels,kernel_size=3,padding = 1))
  layers.append(nn.BatchNorm2d(last_conv_channels))
  layers.append(nn.ReLU())
  drop_layer_last = trial.suggest_float("drop_layer_last",0.1,0.5)
  layers.append(nn.Dropout(drop_layer_last))
  layers.append(nn.Flatten(start_dim=1, end_dim=-1))
  layers.append(nn.Linear(last_conv_channels*224*224, 5))
  #layers.append(nn.LogSoftmax(dim=1))
  return nn.Sequential(*layers)

batch_size = 8
classes = 5
epochs = 15
log_interval = 10
n_train_examples = batch_size * 50
n_valid_examples = batch_size * 20

def objective(trial):
  # Generate the model.
  batch_size = 8
  classes = 5
  epochs = 15
  log_interval = 10
  n_train_examples = batch_size * 50
  n_valid_examples = batch_size * 20
  
  
  image_path = "/content/dataset"
  train_csv = "/content/drive/TRAIN_CSV_PATH/dataset-train.csv"
  test_csv_path = "/content/drive/TRAIN_CSV_PATH/dataset-test.csv"
  basic_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  train_dataset = Cassava_Dataset(image_path,train_csv,basic_transform)
  valid_set = Cassava_Dataset(image_path,test_csv_path,test_transform)
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=batch_size,shuffle=True)
  
  weight_class_0 = trial.suggest_float("weight_class0", 0.5, 5)
  weight_class_1 = trial.suggest_float("weight_class1", 0.5, 5)
  weight_class_2 = trial.suggest_float("weight_class2", 0.5, 5)
  weight_class_3 = trial.suggest_float("weight_class3", 0.5, 5)
  weight_class_4 = trial.suggest_float("weight_class4", 0.5, 5)
  weights = torch.tensor([weight_class_0,weight_class_1,weight_class_2,weight_class_3,weight_class_4]).to(device)
  criterion = nn.CrossEntropyLoss(weight=weights)

  

  model = define_model(trial).to(device)
  # Generate the optimizers.
  lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True) # log=True, will use log scale to interplolate between lr
  optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

  optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
  # alternative version
  # optimizer = trial.suggest_categorical("optimizer", [optim.Adam, optim.RMSprop, optim.SGD])
  # Get the MNIST dataset.
  train_loader, valid_loader = train_loader,valid_loader
  # Training of the model.
  for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      # Limiting training data for faster epochs.
      if batch_idx * batch_size >= n_train_examples:
        break
      data, target = data.data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
 # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(valid_loader):
      # Limiting validation data.
        if batch_idx * batch_size >= n_valid_examples:
          break
        data, target = data.to(device), target.to(device)
        output = model(data)
        # Get the index of the max log-probability.
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)
    # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
    trial.report(accuracy, epoch)
    # then, Optuna can decide if the trial should be pruned
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
  return accuracy

sampler = optuna.samplers.TPESampler()
study = optuna.create_study(study_name="cassava-cnn", direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=500)#, timeout=600)trial,train_loader=train_loader,valid_loader=valid_loader

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print(" Number of finished trials: ", len(study.trials))
print(" Number of pruned trials: ", len(pruned_trials))
print(" Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print(" Value: ", trial.value)

print(" Params: ")
for key, value in trial.params.items():
  print(" {}: {}".format(key, value))


optuna.visualization.plot_param_importances(study)
