import torchvision
import torchvision.transforms as transforms
import torch.utils.data as t_data

from datasets.CUB_200_2011 import CUB_200_2011
from datasets.TinyImageNet import TinyImageNet

# Loading the CIFAR10 training datasets and applying transformation
def cifar10_training_dataset():
    training_transform = transforms.Compose([
        transforms.RandomCrop(24), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4923172, 0.48307145, 0.4474483),(0.24041407,0.23696952,0.25565723),True)
    ])
    training_data = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=training_transform)
    return training_data

# Loading the CIFAR10 test datasets and applying transformation
def cifar10_test_dataset():
    test_transform = transforms.Compose([
        transforms.CenterCrop(24), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4923172, 0.48307145, 0.4474483),(0.24041407,0.23696952,0.25565723),True)
    ])
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)
    return test_data

# Loading the CIFAR10 training dataloader with a given batch_size
def cifar10_training_loader(batch_size: int = 128):
    training_data = cifar10_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the CIFAR10 test dataloader
def cifar10_test_loader():
    test_data = cifar10_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=2)
    return test_loader

# Loading the CIFAR100 training datasets and applying transformation
def cifar100_training_dataset():
    training_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761])
    ])
    training_data = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=training_transform)
    training_data = t_data.Subset(training_data, range(40000))
    return training_data

# Loading the CIFAR100 test datasets and applying transformation (using validation set according to Müller et al.)
def cifar100_test_dataset():
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761])
    ])
    test_data = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=test_transform)
    test_data = t_data.Subset(test_data, range(40000, 50000))
    return test_data
    
# Loading the CIFAR100 training dataloader with a given batch_size
def cifar100_training_loader(batch_size: int = 128):
    training_data = cifar100_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the CIFAR100 test dataloader
def cifar100_test_loader():
    test_data = cifar100_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=2)
    return test_loader

# Loading the MNIST training datasets and applying transformation
def mnist_training_dataset(augmentation: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(0, (0.07143,0.07143))
    ]) if augmentation else transforms.Compose([transforms.ToTensor()])
    training_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return training_data

# Loading the MNIST test datasets and applying transformation
def mnist_test_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return test_data

# Loading the MNIST training dataloader with a given batch_size
def mnist_training_loader(augmentation: bool, batch_size=512):
    training_data = mnist_training_dataset(augmentation)
    training_loader = t_data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the MNIST test dataloader
def mnist_test_loader():
    test_data = mnist_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=2)
    return test_loader

# Loading the FMNIST training datasets and applying transformation
def fmnist_training_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(0, (0.07143,0.07143))
    ])
    training_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    return training_data

# Loading the FMNIST test datasets and applying transformation
def fmnist_test_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return test_data

# Loading the FMNIST training dataloader with a given batch_size
def fmnist_training_loader():
    training_data = fmnist_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=512, shuffle=True, num_workers=2)
    return training_loader

# Loading the FMNIST test dataloader 
def fmnist_test_loader():
    test_data = fmnist_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=2)
    return test_loader

# Loading the EMNIST training datasets and applying transformation
def emnist_training_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(0, (0.07143,0.07143))
    ])
    training_data = torchvision.datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
    return training_data

# Loading the EMNIST test datasets and applying transformation
def emnist_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_data = torchvision.datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)
    return test_data

# Loading the EMNIST training dataloader with a given batch_size
def emnist_training_loader():
    training_data = emnist_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=512, shuffle=True, num_workers=2)
    return training_loader

# Loading the EMNIST test dataloader
def emnist_test_loader():
    test_data = emnist_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=2)
    return test_loader

# Loading the CUB-200-2011 training datasets and applying transformation
def cub_200_2011_training_dataset():
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees = 45,
            translate = (0.0, 0.35),
            scale = (0.65, 1.35)
        ),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    training_data = CUB_200_2011('data/', split="train", transforms=transform)
    return training_data

# Loading the CUB-200-2011 test datasets and applying transformation
def cub_200_2011_test_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_data = CUB_200_2011('data/', split="test", transforms=transform)
    return test_data

# Loading the CUB-200-2011 training dataloader with a given batch_size
def cub_200_2011_training_loader(batch_size = 256):
    training_data = cub_200_2011_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the CUB-200-2011 test dataloader with a given batch_size
def cub_200_2011_test_loader(batch_size = 256):
    test_data = cub_200_2011_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size, shuffle=True, num_workers=2)
    return test_loader

# Loading the TinyImageNet training datasets and applying transformation
def tinyimagenet_training_dataset():
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees = 7.5,
            translate = (0.0, 0.1),
            scale = (0.925, 1.075)
        ),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    training_data = TinyImageNet('data/', split="train", transforms=transform)
    return training_data

# Loading the TinyImageNet test datasets and applying transformation
def tinyimagenet_test_dataset():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    test_data = TinyImageNet('data/', split="test", transforms=transform)
    return test_data

# Loading the TinyImageNet training dataloader with a given batch_size
def tinyimagenet_training_loader(batch_size = 128):
    training_data = tinyimagenet_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the TinyImageNet test dataloader with a given batch_size
def tinyimagenet_test_loader(batch_size = 256):
    test_data = tinyimagenet_test_dataset()
    test_loader = t_data.DataLoader(test_data, batch_size, shuffle=True, num_workers=2)
    return test_loader