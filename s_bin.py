#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import matplotlib


# In[3]:


project_name = 'Model_SmartBin'


# In[4]:


os.getcwd()


# In[5]:


data_dir ='./Downloads/s_bin/dataset'
print(os.listdir(data_dir))


# In[7]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# In[8]:


# Define the data transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the input images to the size expected by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet statistics
])

# Define your dataset paths (adjust these paths according to your dataset)
train_dataset = ImageFolder(os.path.join(data_dir,'train'), transform=transform)
val_dataset = ImageFolder(os.path.join(data_dir,'val'), transform=transform)
test_dataset = ImageFolder(os.path.join(data_dir,'test'), transform=transform)


# In[10]:


# Create data loaders
batch_size = 128
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size)
test_dl = DataLoader(test_dataset, batch_size=batch_size)

# Load a pretrained ResNet model (e.g., ResNet-50)
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Modify the classification head for your specific task (6 classes)
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[6]:


from torchvision.models import resnet50, ResNet50_Weights


# In[12]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress tracking


# Training loop with progress bar
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Use tqdm to create a progress bar
    pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    
    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Update progress bar description
        pbar.set_description(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / (total_samples / batch_size):.4f} - Acc: {100 * correct_predictions / total_samples:.2f}%')

    # Validation loop (similar to the previous example)
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in val_dl:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {validation_accuracy:.2f}%')

# Test the model on the test set and evaluate its performance
model.eval()



# In[13]:


save_directory ='/Users/maniksinghsarmaal/Downloads/s_bin/trained_model'
torch.save(model.state_dict(), save_directory +'waste_classification_model.pth')


# In[14]:


with torch.no_grad():
    correct = 0
    total = 0
    
    for images, labels in test_dl:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')


# In[15]:


import jovian
jovian.commit()


# In[20]:


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# In[23]:


image_path = "/Users/maniksinghsarmaal/Downloads/test1.jpeg"
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0) 

with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
# Print the predicted class
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
predicted_label = class_names[predicted_class]
plt.imshow(image)
print(f"Predicted class: {predicted_label}")


# In[ ]:




