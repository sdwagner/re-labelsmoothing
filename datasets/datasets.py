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

# Loading the CIFAR10 validation datasets and applying transformation
def cifar10_validation_dataset():
    test_transform = transforms.Compose([
        transforms.CenterCrop(24), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4923172, 0.48307145, 0.4474483),(0.24041407,0.23696952,0.25565723),True)
    ])
    validation_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)
    return validation_data

# Loading the CIFAR10 training dataloader with a given batch_size
def cifar10_training_loader(batch_size: int = 128):
    training_data = cifar10_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the CIFAR10 validation dataloader
def cifar10_validation_loader():
    validation_data = cifar10_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=True, num_workers=2)
    return validation_loader

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

# Loading the CIFAR100 validation datasets and applying transformation
def cifar100_validation_dataset():
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761])
    ])
    validation_data = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=test_transform)
    validation_data = t_data.Subset(validation_data, range(40000, 50000))
    return validation_data
    
# Loading the CIFAR100 training dataloader with a given batch_size
def cifar100_training_loader(batch_size: int = 128):
    training_data = cifar100_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the CIFAR10 validation dataloader
def cifar100_validation_loader():
    validation_data = cifar100_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=True, num_workers=2)
    return validation_loader

# Loading the MNIST training datasets and applying transformation
def mnist_training_dataset(augmentation: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(0, (0.07143,0.07143))
    ]) if augmentation else transforms.Compose([transforms.ToTensor()])
    training_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return training_data

# Loading the MNIST validation datasets and applying transformation
def mnist_validation_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    validation_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return validation_data

# Loading the MNIST training dataloader with a given batch_size
def mnist_training_loader(augmentation: bool, batch_size=512):
    training_data = mnist_training_dataset(augmentation)
    training_loader = t_data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the MNIST validation dataloader
def mnist_validation_loader():
    validation_data = mnist_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=True, num_workers=2)
    return validation_loader

# Loading the FMNIST training datasets and applying transformation
def fmnist_training_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(0, (0.07143,0.07143))
    ])
    training_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    return training_data

# Loading the FMNIST validation datasets and applying transformation
def fmnist_validation_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    validation_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return validation_data

# Loading the FMNIST training dataloader with a given batch_size
def fmnist_training_loader():
    training_data = fmnist_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=512, shuffle=True, num_workers=2)
    return training_loader

# Loading the FMNIST validation dataloader 
def fmnist_validation_loader():
    validation_data = fmnist_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=True, num_workers=2)
    return validation_loader

# Loading the EMNIST training datasets and applying transformation
def emnist_training_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(0, (0.07143,0.07143))
    ])
    training_data = torchvision.datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
    return training_data

# Loading the EMNIST validation datasets and applying transformation
def emnist_validation_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    validation_data = torchvision.datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)
    return validation_data

# Loading the EMNIST training dataloader with a given batch_size
def emnist_training_loader():
    training_data = emnist_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size=512, shuffle=True, num_workers=2)
    return training_loader

# Loading the EMNIST validation dataloader
def emnist_validation_loader():
    validation_data = emnist_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=True, num_workers=2)
    return validation_loader

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

# Loading the CUB-200-2011 validation datasets and applying transformation
def cub_200_2011_validation_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    validation_data = CUB_200_2011('data/', split="val", transforms=transform)
    return validation_data

# Loading the CUB-200-2011 training dataloader with a given batch_size
def cub_200_2011_training_loader(batch_size = 256):
    training_data = cub_200_2011_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the CUB-200-2011 validation dataloader with a given batch_size
def cub_200_2011_validation_loader(batch_size = 256):
    validation_data = cub_200_2011_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size, shuffle=True, num_workers=2)
    return validation_loader

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

# Loading the TinyImageNet validation datasets and applying transformation
def tinyimagenet_validation_dataset():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    validation_data = TinyImageNet('data/', split="val", transforms=transform)
    return validation_data

# Loading the TinyImageNet training dataloader with a given batch_size
def tinyimagenet_training_loader(batch_size = 128):
    training_data = tinyimagenet_training_dataset()
    training_loader = t_data.DataLoader(training_data, batch_size, shuffle=True, num_workers=2)
    return training_loader

# Loading the TinyImageNet validation dataloader with a given batch_size
def tinyimagenet_validation_loader(batch_size = 256):
    validation_data = tinyimagenet_validation_dataset()
    validation_loader = t_data.DataLoader(validation_data, batch_size, shuffle=True, num_workers=2)
    return validation_loader