from models import Generator
import torch

# Load the trained generator
generator = Generator()
generator.load_state_dict(torch.load('generator_state.pth'))

# Function to generate a caption for an image
def generate_caption(image):
    output = generator(image.unsqueeze(0))
    # Process output to generate caption
    return "Generated caption based on the image."

# Sample usage example
# image = load_single_image('path/to/an/image.jpg')
# print(generate_caption(image))
