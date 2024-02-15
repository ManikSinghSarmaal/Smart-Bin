import torch
import torchvision
input_shape = 224
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

#load the model .pth file
model_save_path = "/Users/maniksinghsarmaal/Downloads/s_bin/s_bin_best/pth file/mobilenetv2_model.pth"
# Load the model
loaded_model = torchvision.models.mobilenet_v2(pretrained=False)

# Modify the last layer
number_features = loaded_model.classifier[1].in_features
features = list(loaded_model.classifier.children())[:-1]  # Remove last layer
features.extend([torch.nn.Linear(number_features, 6)])  # Set the correct number of classes
loaded_model.classifier = torch.nn.Sequential(*features)

# Load the state dictionary
loaded_model.load_state_dict(torch.load(model_save_path))

# Set the model to evaluation mode
loaded_model.eval()


#pre-process the image
from torchvision import transforms
from PIL import Image

# Load and preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # Replace with your actual class labels

with torch.no_grad():
    # Make predictions
    output = loaded_model(preprocess_image("/Users/maniksinghsarmaal/Downloads/s_bin/dataset/test/cardboard/cardboard84.jpg"))

    # Get class predictions
    _, predicted_class = torch.max(output, 1)

    # Print or use the predicted class index and label as needed
    predicted_index = predicted_class.item()
    predicted_label = class_labels[predicted_index]
    print("Predicted Class Index:", predicted_index)
    print("Predicted Class Label:", predicted_label)