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

#print(mobilenet_v2_based)

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(mobilenet_v2_based.parameters(), lr=0.001, momentum=0.9)
def train_model(model, criterion, optimizer, num_epochs=25):
   since = time.time()

   for epoch in range(num_epochs):
       print('Epoch {}/{}'.format(epoch, num_epochs - 1))
       print('-' * 10)

       #set model to trainable
       # model.train()

       train_loss = 0

       # Iterate over data.
       for i, data in enumerate(dataloaders['train']):
           inputs , labels = data
           inputs = inputs.to(device)
           labels = labels.to(device)

           optimizer.zero_grad()
          
           with torch.set_grad_enabled(True):
               outputs  = model(inputs)
               loss = criterion(outputs, labels)

           loss.backward()
           optimizer.step()

           train_loss += loss.item() * inputs.size(0)

           print('{} Loss: {:.4f}'.format(
               'train', train_loss / dataset_sizes['train']))
          
   time_elapsed = time.time() - since
   print('Training complete in {:.0f}m {:.0f}s'.format(
       time_elapsed // 60, time_elapsed % 60))

   return model

def visualize_model(model, num_images=6):
   was_training = model.training
   model.eval()
   images_so_far = 0
   fig = plt.figure()

   with torch.no_grad():
       for i, (inputs, labels) in enumerate(dataloaders['validation']):
           inputs = inputs.to(device)
           labels = labels.to(device)

           outputs = model(inputs)
           _, preds = torch.max(outputs, 1)

           for j in range(inputs.size()[0]):
               images_so_far += 1
               ax = plt.subplot(num_images//2, 2, images_so_far)
               ax.axis('off')
               ax.set_title('predicted: {} truth: {}'.format(class_names[preds[j]], class_names[labels[j]]))
               img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
               img = std * img + mean
               ax.imshow(img)

               if images_so_far == num_images:
                   model.train(mode=was_training)
                   return
       model.train(mode=was_training)

mobilenet_v2_based = train_model(mobilenet_v2_based, criterion, optimizer_ft, num_epochs=25)

visualize_model(mobilenet_v2_based)

# <----> Training complete in 19m 7s  <----->
# VISUALISING AFTER TRAINING PROCESSS !!!

from PIL import Image

def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = data_transforms['val']
    preprocessed_image = transform(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)  # Add batch dimension
    return preprocessed_image

def predict_image(model, image_path):
    model.eval()  # Set model to evaluation mode
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = preprocessed_image.to(device)

    with torch.no_grad():
        output = model(preprocessed_image)
        _, predicted_class = torch.max(output, 1)

    predicted_class = predicted_class.item()
    predicted_label = class_names[predicted_class]

    return predicted_class, predicted_label

# Specify the path to the image you want to test
image_path_to_test = "/Users/maniksinghsarmaal/Downloads/s_bin/dataset/test/plastic/plastic404.jpg"

# Call the predict_image function
predicted_class, predicted_label = predict_image(mobilenet_v2_based, image_path_to_test)

# Print the result
print(f"Predicted Class: {predicted_class}")
print(f"Predicted Label: {predicted_label}")