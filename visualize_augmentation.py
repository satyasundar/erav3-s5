import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset with augmentation
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)

# Function to denormalize images for visualization
def denormalize(tensor):
    tensor = tensor * 0.3081 + 0.1307
    return torch.clamp(tensor, 0, 1)

# Get a sample image
def plot_augmentations():
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    
    # Get one image
    original_image, _ = train_dataset[0]
    
    # Show original
    axs[0, 0].imshow(denormalize(original_image[0]), cmap='gray')
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')
    
    # Show different augmentations
    for i in range(1, 10):
        augmented_image, _ = train_dataset[0]  # Get new augmentation of same image
        ax = axs[i//5, i%5]
        ax.imshow(denormalize(augmented_image[0]), cmap='gray')
        ax.set_title(f'Augmentation {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png')
    plt.close()

if __name__ == "__main__":
    plot_augmentations() 