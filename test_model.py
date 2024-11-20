import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import MNISTModel
import pytest
import glob

def get_latest_model():
    model_files = glob.glob('mnist_model_*.pth')
    return max(model_files, key=os.path.getctime)

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_batch_processing():
    model = MNISTModel()
    # Test batch processing
    batch_size = 32
    test_batch = torch.randn(batch_size, 1, 28, 28)
    output = model(test_batch)
    assert output.shape == (batch_size, 10), f"Batch output shape should be ({batch_size}, 10)"

def test_model_components():
    model = MNISTModel()
    
    # Test if model has expected layers
    assert hasattr(model, 'conv1'), "Model should have conv1 layer"
    assert hasattr(model, 'conv2'), "Model should have conv2 layer"
    assert hasattr(model, 'fc1'), "Model should have fc1 layer"
    assert hasattr(model, 'fc2'), "Model should have fc2 layer"
    
    # Test conv1 specifications
    assert model.conv1.in_channels == 1, "Conv1 should have 1 input channel"
    assert model.conv1.out_channels == 8, "Conv1 should have 8 output channels"
    
    # Test final layer
    assert model.fc2.out_features == 10, "Final layer should have 10 outputs"

def test_model_output_range():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Test if output values are reasonable (before softmax)
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains infinite values"

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be above 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 