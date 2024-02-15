from __future__ import print_function, division
import os
import time
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/Users/maniksinghsarmaal/Downloads/s_bin/dataset_copy"
input_shape = 224
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

#data transformation
data_transforms = {
   'train': transforms.Compose([
       transforms.CenterCrop(input_shape),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ]),
   'val': transforms.Compose([
       transforms.CenterCrop(input_shape),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ]),
}

image_datasets = {
   x: datasets.ImageFolder(
       os.path.join(data_dir, x),
       transform=data_transforms[x]
   )
   for x in ['train', 'val']
}

dataloaders = {
   x: torch.utils.data.DataLoader(
       image_datasets[x], batch_size=32,
       shuffle=True, num_workers=4
   )
   for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

#print(dataset_sizes)
class_names = image_datasets['train'].classes
#print(class_names)
device = torch.device("mps")
#print(device)## Load the model based on VGG19
# Load the model based on MobileNetV2
mobilenet_v2_based = torchvision.models.mobilenet_v2(pretrained=True)

# Freeze the layers
for param in mobilenet_v2_based.parameters():
   param.requires_grad = False

# Modify the last layer
number_features = mobilenet_v2_based.classifier[1].in_features
features = list(mobilenet_v2_based.classifier.children())[:-1]  # Remove last layer
features.extend([torch.nn.Linear(number_features, len(class_names))])
mobilenet_v2_based.classifier = torch.nn.Sequential(*features)

mobilenet_v2_based = mobilenet_v2_based.to(device)

print(mobilenet_v2_based)

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(mobilenet_v2_based.parameters(), lr=0.001, momentum=0.9)


mobilenet_v2_based = mobilenet_v2_based.to(device)

print(mobilenet_v2_based)

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(mobilenet_v2_based.parameters(), lr=0.001, momentum=0.9)