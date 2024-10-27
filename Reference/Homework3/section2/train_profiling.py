import wandb
import os
from utils import set_seed
import random
import torch
from torch import nn, optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import argparse
from torchvision.models import resnet18
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import time  # Import time for tracking execution time
from torch.profiler import profile, record_function, ProfilerActivity

def get_scheduler(use_scheduler, optimizer, **kwargs):
    """
    :param use_scheduler: whether to use lr scheduler
    :param optimizer: instance of optimizer
    :param kwargs: other args to pass to scheduler; already filled with some default values in train_model()
    :return: scheduler
    """
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    if use_scheduler:
        # Properly implement the OneCycleLR scheduler
        # Assuming `max_lr`, `steps_per_epoch`, and `epochs` are passed in as keyword arguments
        max_lr = kwargs.get('max_lr', 0.01)  # Default max_lr if not specified
        total_steps_1 = kwargs.get('total_steps')  # Total steps must be provided or calculated
        if not total_steps_1:
            steps_per_epoch = kwargs.get('steps_per_epoch', 100)  # Default or passed number of steps per epoch
            epochs = kwargs.get('epochs', 10)  # Default or passed number of training epochs
            total_steps_1 = steps_per_epoch * epochs
        anneal_strategy = kwargs.get('anneal_strategy', 'cos')  # Default annealing strategy
        final_div_factor = kwargs.get('final_div_factor', 1e4)  # How much to reduce the lr at the end

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps_1,
            # epochs=epochs,
            pct_start=0.3,  # Percentage of the cycle spent increasing the learning rate
            anneal_strategy=anneal_strategy,
            final_div_factor=final_div_factor
        )
    else:
        scheduler = None

    return scheduler

def scale_learning_rate_and_hyperparameters(learning_rate, batch_size, original_batch_size, beta1, beta2, epsilon):
    """
    Scale the learning rate and other hyperparameters according to the scaling rule for Adam.
    :param learning_rate: Original learning rate
    :param batch_size: New batch size
    :param original_batch_size: Original batch size
    :param beta1: Original beta1 parameter for Adam
    :param beta2: Original beta2 parameter for Adam
    :param epsilon: Original epsilon parameter for Adam
    :return: Scaled learning rate, beta1, beta2, epsilon
    """
    # Calculate the scaling factor
    kappa = batch_size / original_batch_size
    sqrt_kappa = kappa ** 0.5

    # Scale the learning rate
    scaled_lr = learning_rate * sqrt_kappa

    # Scale the beta parameters and epsilon
    scaled_beta1 = 1 - kappa * (1 - beta1)
    scaled_beta2 = 1 - kappa * (1 - beta2)
    scaled_epsilon = epsilon / sqrt_kappa

    return scaled_lr, scaled_beta1, scaled_beta2, scaled_epsilon

def evaluate(model, data_loader, device):
    """
    :param model: instance of model  
    :param data_loader: instance of data loader
    :param device: cpu or cuda
    :return: accuracy, total cross-entropy loss, and list of losses per step (batch)
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')  # Loss summed over the batch

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()  # Accumulate total loss

            _, predicted = torch.max(outputs, 1)


            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0) 

    accuracy = correct_predictions / total_predictions

    model.train()

    # Return accuracy, total loss, and the list of loss values per step
    return accuracy, total_loss, loss.item()

# Define custom Dataset class
class OxfordPetsDataset(Dataset):
    def __init__(self, dataframe, split, label_map, transform=None):
        """
        Args:
        - dataframe (pd.DataFrame): The full dataframe containing image paths, labels, and splits.
        - split (str): The split to use ('train', 'val', or 'test').
        - label_map (dict): Predefined label map to ensure consistency across splits.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Filter the dataframe to only include the specified split
        self.dataframe = dataframe[dataframe['split'] == split]
        self.transform = transform
        self.label_map = label_map  # Use the globally consistent label map

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        img_path = f"Homework1/data/images/{img_name}"  # Adjust path accordingly
        label = self.dataframe.iloc[idx]['label']

        # Convert label to its corresponding integer value
        label = self.label_map[label]

        image = Image.open(img_path).convert('RGB')  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

import time  # Import time for tracking stages
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def train_model(
    run_name,
    model,
    batch_size,
    epochs,
    learning_rate,
    device,
    save_dir,
    use_scheduler,
    original_batch_size=None,
    scale_lr=False,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):
    # Ensure WandB is initialized before logging
    wandb.init(
        project="oxford_pet_classification_PROFILING",
        name=f"lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "scheduler": use_scheduler,
            "model_architecture": "ResNet-18"
        }
    )

    model.to(device)

    # Start total runtime tracking
    total_runtime_start = time.time()

    # Adjust learning rate if needed
    if scale_lr and batch_size:
        learning_rate, beta1, beta2, epsilon = scale_learning_rate_and_hyperparameters(
            learning_rate, batch_size, original_batch_size, beta1, beta2, epsilon
        )

    # Load dataset and DataLoader
    df = pd.read_csv('Homework1/problem_1/oxford_pet_split.csv')
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_set = OxfordPetsDataset(df, 'train', label_map, transform)
    val_set = OxfordPetsDataset(df, 'val', label_map, transform)
    test_set = OxfordPetsDataset(df, 'test', label_map, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    scheduler = get_scheduler(use_scheduler, optimizer, max_lr=learning_rate, total_steps=epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(save_dir, f'{run_name}_checkpoint.pth')  # Simplified checkpoint path

    # Initialize timers
    data_loading_time = 0
    forward_pass_time = 0
    backward_pass_time = 0
    evaluation_time = 0

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}")

        model.train()
        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch}", unit='img') as pbar:
            for inputs, labels in train_loader:
                # Measure data loading time
                start_time = time.time()
                inputs, labels = inputs.to(device), labels.to(device)
                data_loading_time += time.time() - start_time

                # Measure forward pass time
                start_time = time.time()
                outputs = model(inputs)
                forward_pass_time += time.time() - start_time

                # Measure backward pass time (loss calculation + backpropagation)
                start_time = time.time()
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_pass_time += time.time() - start_time

                if use_scheduler:
                    scheduler.step()

                pbar.update(inputs.size(0))
                global_step += 1

                # Evaluate and checkpoint every epoch
                if global_step % (len(train_loader) // batch_size) == 0:
                    # Measure evaluation time
                    start_time = time.time()
                    val_acc, val_loss, _ = evaluate(model, val_loader, device)
                    evaluation_time += time.time() - start_time

                    # Save the best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Checkpoint saved at: {checkpoint_path}")

                    # Log metrics to WandB
                    wandb.log({
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "global_step": global_step
                    })

                pbar.set_postfix(loss=loss.item())

    # Load the saved checkpoint (if it exists)
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path))

    # Measure test evaluation time
    start_time = time.time()
    test_acc, test_loss, _ = evaluate(model, test_loader, device)
    evaluation_time += time.time() - start_time

    print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

    # Log final test results and timings to WandB
    total_runtime = time.time() - total_runtime_start
    other_time = total_runtime - (data_loading_time + forward_pass_time + backward_pass_time + evaluation_time)

    wandb.log({
        "final_test_accuracy": test_acc,
        "final_test_loss": test_loss,
        "total_runtime": total_runtime,
        "data_loading_time": data_loading_time,
        "forward_pass_time": forward_pass_time,
        "backward_pass_time": backward_pass_time,
        "evaluation_time": evaluation_time,
        "other_time": other_time
    })

    # Print all timing results
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Data loading time: {data_loading_time:.2f} seconds")
    print(f"Forward pass time: {forward_pass_time:.2f} seconds")
    print(f"Backward pass time: {backward_pass_time:.2f} seconds")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    print(f"Other time: {other_time:.2f} seconds")

    wandb.finish()

# Define the sweep configuration
sweep_config = {
    'method': 'grid',  # Using grid method for fixed learning rates
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'  # Objective is to maximize validation accuracy
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-2, 1e-4, 1e-5, 1e-3]  # Learning rates to test
        },
        'batch_size': {
            'values': [32]
        },
        'epochs': {
            'values': [5]
        },
        'use_scheduler': {
            # 'values': [True, False]
            'values': [False]
        }
    }
}

def sweep_train():
    # Initialize a wandb run for this sweep iteration
    with wandb.init() as run:
        set_seed(42)
        config = run.config  # Access the configuration for this sweep iteration

        # Set up the training environment based on the config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet18(pretrained=False, num_classes=37).to(device)

        # Call the existing training function with parameters from the config
        train_model(
            run_name=run.name,
            model=model,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device,
            save_dir='./checkpoints',
            use_scheduler=False
        )

def get_args():
    parser = argparse.ArgumentParser(description='E2EDL training script')
    # exp description
    parser.add_argument('--run_name', type=str, default='baseline',
                        help="a brief description of the experiment; "
                             "alternatively, you can set the name automatically based on hyperparameters:"
                             "run_name = f'lr_{learning_rate}_bs_{batch_size}...' to reflect key hyperparameters")
    # dirs
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='save best checkpoint to this dir')
    # training config
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size; modify this to fit your GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_scheduler', action='store_true', help='use lr scheduler')

    # IMPORTANT: if you are copying this script to notebook, replace 'return parser.parse_args()' with 'args = parser.parse_args("")'

    return parser.parse_args()

if __name__ == '__main__':
    
    rid = random.randint(0, 1000000)
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=False, num_classes=37).to(device)
    train_model(
        run_name=args.run_name,
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        use_scheduler=False,
    )
