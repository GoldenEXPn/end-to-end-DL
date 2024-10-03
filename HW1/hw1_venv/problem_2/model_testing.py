import torch
import torch.nn.functional as F
from contextlib import contextmanager


# functions used for pre-train testing

# Data Leakage Check
## use the returned datasets from get_dataset() to check for data leakage, instead of using the source code
'''from utils.datasets import get_dataset, check_leakage_using_hash
train_dataset, val_dataset, test_dataset = get_dataset()'''
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
'''from torch.utils.data import DataLoader
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

# Brightness Test
from torch.utils.data import DataLoader
from utils.datasets import get_testset
from utils.trained_models import get_trained_model
test_dataset = get_testset()
test_loader = DataLoader(test_dataset, batch_size=32)
trained_model = get_trained_model()




























