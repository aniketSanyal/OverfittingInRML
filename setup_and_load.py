import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration - GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
mnist_train_loader = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(dataset=mnist_test, batch_size=64, shuffle=False)

# CIFAR10 dataset
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
cifar_train_loader = DataLoader(dataset=cifar_train, batch_size=64, shuffle=True)
cifar_test_loader = DataLoader(dataset=cifar_test, batch_size=64, shuffle=False)
