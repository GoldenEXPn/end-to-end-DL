
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Define data augmentation transformations for training and validation
transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def get_cifar10_classes(train_transforms, test_transforms):
    # Load CIFAR-10 dataset with data augmentation for training, no augmentation for validation
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=train_transforms)
    val_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=test_transforms)
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=test_transforms)

    # Class mapping
    class_mapping = {3: 0, 5: 1}  # 'cat': 0, 'dog': 1

    # Filter the datasets to only include 'cat' and 'dog' classes and remap their labels
    train_data = [(img, class_mapping[label]) for img, label in train_data if label in class_mapping]
    val_data = [(img, class_mapping[label]) for img, label in val_data if label in class_mapping]
    test_data = [(img, class_mapping[label]) for img, label in test_data if label in class_mapping]

    # Split training data into training and validation sets
    train_indices, val_indices = train_test_split(
        list(range(len(train_data))), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(train_data, train_indices)
    val_dataset = torch.utils.data.Subset(val_data, val_indices)

    return train_dataset, val_dataset, test_data
