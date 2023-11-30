import torch
import torchvision
from torchvision import transforms

class DatasetsLoader():
    def __init__(self, batch_size_train=32, batch_size_test=256, dataset_name = 'CIFAR-10', download = False, pin_memory = False, shuffle_train = True, num_workers=2):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        if dataset_name == 'CIFAR-10':
            #CIFAR-10 dataset
            # Downloading/Loading CIFAR10 data
            self.trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=download, transform=transform)
            self.testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=download, transform=transform)
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            raise ValueError("Unsupported dataset name. Choose 'CIFAR-10'.")
        
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size_train, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size_test * 4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)      

    def load_data(self):
        """
        Returns:
        torch.utils.data.DataLoader trainloader.
        torch.utils.data.DataLoader testloader.
        tuple classes: List of classes in Dataset.
        """
        return self.trainloader, self.testloader, self.classes