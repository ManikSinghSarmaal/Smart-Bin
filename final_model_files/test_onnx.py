import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Define the path to the ONNX model file
onnx_model_path = "model.onnx"

# Define the transformation for test images
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

class_labels = ['Plastic','Organic','Metal']
# Function to predict using the ONNX model
def predict_image(image_path, model_path):
    # Load the ONNX model
    ort_session = onnxruntime.InferenceSession(model_path)

    # Get the input name expected by the model
    input_name = ort_session.get_inputs()[0].name

    # Preprocess the input image
    input_data = preprocess_image(image_path)

    # Run inference using ONNX runtime
    output = ort_session.run(None, {input_name: input_data.astype(np.float32)})

    # Get the predicted class index
    predicted_class_index = np.argmax(output)
    
    return predicted_class_index
    
# Example usage: Predict a new image
image_path = "/Users/maniksinghsarmaal/Downloads/ouch.jpg"
predictions = predict_image(image_path, onnx_model_path)
print("Predicted Class:", class_labels[predictions])