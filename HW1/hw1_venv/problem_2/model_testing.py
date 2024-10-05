# functions used for pre-train testing

# Data Leakage Check
## use the returned datasets from get_dataset() to check for data leakage, instead of using the source code
'''from utils.datasets import get_dataset, check_leakage_using_hash
train_dataset, val_dataset, test_dataset = get_dataset()'''
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2.functional import rgb_to_grayscale_image

# check_leakage_using_hash(train_dataset, val_dataset, test_dataset)



# Model Architecture Check / Gradient Descent Validation
'''from utils.models import get_model
model = get_model()'''
# Model Architecture Check
'''
num_classes = 10
input = torch.randn(4, 3, 32, 32)
output = model(input)
expected_output = (4, num_classes)
print(f"Got output shape: {output.shape}, expected output shape: {expected_output}")
'''
# Gradient Descent Validation
'''
train_loader = DataLoader(train_dataset, batch_size=32)
inputs, labels = next(iter(train_loader))
outputs = model(inputs)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
all_updated = True
non_count, total=0,0
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        non_count += 1
        print(f'Parameter \"{name}\" is not updated.')
        all_updated = False
    total+=1
if all_updated: print("All parameters are updated.")
else: print(f"Summary: {non_count}/{total} parameters are not updated.")
'''


# Learning Rate Check:
# These steps provide necessary components for learning rate range test for torch_lr_finder.LRFinder
'''
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
optimizer = AdamW(model.parameters(), lr=1e-6) # the lr is set to 1e-6 as specified here
criterion = CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot(log_lr=False)
lr_finder.reset()
'''


# Dying ReLU Examination
# Model Robustness Test


## Check Dying ReLU
'''
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from torch.utils.data import DataLoader
from utils.datasets import get_testset
from utils.trained_models import get_trained_model, get_summary

# Load test data
test_dataset = get_testset()
test_loader = DataLoader(test_dataset, batch_size=32)
relu_stats = []
original_relu = F.relu
def custom_relu(X):
    y = original_relu(X)
    num_zeros = torch.sum(y == 0).item()
    total_neurons = y.numel()
    dead_percentage = num_zeros / total_neurons * 100
    relu_stats.append(dead_percentage)
    return y
# Help to use with statement
@contextmanager
def capture_relu():
    global original_relu
    original_relu = F.relu
    F.relu = custom_relu
    try:
        yield
    finally:
        F.relu = original_relu  # Restore the original F.relu after capturing

def check_dying_relu(trained_model, data_loader):
    # Reset relu_stats
    global relu_stats
    relu_stats = []
    trained_model.eval()
    with torch.no_grad():
        with capture_relu():  # Capture F.relu and replace with custom relu for inspection
            for inputs, _ in data_loader:
                trained_model(inputs)
                break
    for i, stat in enumerate(relu_stats):
        print(f"ReLU layer {i + 1}: {stat:.2f}% of neurons are dead.")
trained_model = get_trained_model()
check_dying_relu(trained_model, test_loader)
# Give summary
get_summary(relu_stats)'''

## Model Robustness
import torch
import wandb
from utils.datasets import get_testset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision.transforms import transforms
from utils.trained_models import get_trained_model
from torchvision.transforms import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_trained_model().to(device)

def evaluate(model, data_loader, device):
    model.eval()
    cor_preds=0
    total_preds=0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            cor_preds += torch.sum(preds == labels.data).item()
    return cor_preds / total_preds


# Brightness Test
lambda_factors = [0.2, 0.4, 0.6, 0.8, 1.0]
result = {}


'''
    wandb.init(project="277 Robustness Test", name="brightness_test")
    for factor in lambda_factors:
    test_dataset = get_testset()
    brightness_transform = transforms.Compose([
        transforms.Lambda(lambda img: F.adjust_brightness(img, factor)),
        *test_dataset.dataset.transform.transforms
    ])
    test_dataset.dataset.transform = brightness_transform
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate(model, test_loader, device)
    wandb.log({'lambda_factor': factor, 'accuracy': accuracy})
    print(f'lambda_factor: {factor}, accuracy: {accuracy}')'''


# Rotation Test
'''wandb.init(project="277 Robustness Test", name="rotation_test")
for angle in range(0, 360, 60):
    test_dataset = get_testset()
    rotation_transform = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle)),
        *test_dataset.dataset.transform.transforms
    ])
    test_dataset.dataset.transform = rotation_transform
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate(model, test_loader, device)
    wandb.log({'rotation_angle': angle, 'accuracy': accuracy})
    print(f'rotation_angle: {angle}, accuracy: {accuracy}')'''


# Normalization Mismatch
import numpy as np
from PIL import Image

def calculate_stats(dataset):
    rgb_values_list = []
    for img,_ in dataset:
        img = transforms.ToPILImage()(img)
        rgb_values_list.append(np.asarray(img).reshape(-1, 3)) # reshape to rgb channel
    rgb_values = np.concatenate(rgb_values_list, axis=0) / 255.0
    return np.mean(rgb_values, axis=0), np.std(rgb_values, axis=0)

# Calculate test set stats
test_dataset = get_testset()
mu_test, mu_std = calculate_stats(test_dataset)

# Calculate train set stats
train_set = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=0)
data = next(iter(train_loader))
images, labels = data
mu_train = images.mean([0,2,3])
std_train = images.std([0,2,3])

# Examine differences
print(f'Testing mean: {mu_test}, std: {mu_std}')
print(f'Testing mean: {mu_train}, std: {std_train}')








