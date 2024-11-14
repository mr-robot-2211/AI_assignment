# test.py
import torch
import torchvision
import torchvision.transforms as transforms
from model import ForegroundFeatureAveragingWithAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 test loader
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Load model
model = ForegroundFeatureAveragingWithAttention(device).to(device)
model.eval()

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
