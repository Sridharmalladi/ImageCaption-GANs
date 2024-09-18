import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Function to load images from a directory
def load_images(directory, target_size=(256, 256)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.ANTIALIAS)
            images.append(img)
    return images

# Function to apply transformations to images
def transform_images(images):
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return [tensor_transform(image) for image in images]
