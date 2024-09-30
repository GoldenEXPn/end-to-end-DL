import torch
from PIL import Image
import imagehash
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Define data augmentation transformations for training and validation
transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# Download CIFAR-10; split into train/val/test with fixed seed
def get_dataset():
    # Load CIFAR-10 dataset with data augmentation for training, no augmentation for validation
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_no_aug)
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_no_aug)

    val_indices, test_indices = train_test_split(
        list(range(len(test_data))), test_size=0.5, random_state=42)

    val_dataset = torch.utils.data.Subset(test_data, val_indices)
    test_dataset = torch.utils.data.Subset(test_data, test_indices)
    train_dataset = torch.utils.data.ConcatDataset([train_data, test_dataset])

    return train_dataset, val_dataset, test_dataset


def get_testset():
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_no_aug)
    test_indices, _ = train_test_split(
        list(range(len(test_data))), test_size=0.5, random_state=42)
    test_dataset = torch.utils.data.Subset(test_data, test_indices)
    return test_dataset


def hash_image(image):
    if isinstance(image, Image.Image):
        return imagehash.phash(image)
    else:
        image = image.detach().numpy()
        image = image.transpose(1, 2, 0)
        image = (image * 255).astype('uint8')
        image = Image.fromarray(image)
        return imagehash.phash(image)


def check_leakage_using_hash(train_dataset, val_dataset, test_dataset):
    train_hashes = {hash_image(img): idx for idx, (img, label) in enumerate(train_dataset)}
    val_hashes = {hash_image(img): idx for idx, (img, label) in enumerate(val_dataset)}
    test_hashes = {hash_image(img): idx for idx, (img, label) in enumerate(test_dataset)}

    leakage=False  # initialize leakage flag
    if train_hashes.keys() & val_hashes.keys():
        leakage=True
        print('Data leakage found between train set and val set')
    if train_hashes.keys() & test_hashes.keys():
        leakage=True
        print('Data leakage found between train set and test set')
    if val_hashes.keys() & test_hashes.keys():
        leakage=True
        print('Data leakage found between val set and test set')
    if not leakage:
        print('No leakage found')





















