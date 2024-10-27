import os
import math
import torch
import wandb
import random
import argparse
import pandas as pd
from spacy.cli.train import train
from torch.xpu import device
from tqdm import tqdm
from PIL import Image
from utils import set_seed
from torch import nn, optim
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
from torch.profiler import profile, record_function, ProfilerActivity

# ... [rest of your imports and existing code] ...

def train_model(
        run_name,
        model,
        batch_size,
        epochs,
        learning_rate,
        device,
        save_dir,
        use_scheduler,
        rid,
        # For scaling learning rate
        scaled_train=False,
        old_batch_size=None,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
):
    model.to(device)

    # Load dataset and csv file
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # Use ImageNet
    dataset = datasets.OxfordIIITPet(root='data', download=True, transform=transform)
    df = pd.read_csv('oxford_pet_split.csv')
    # Image-label mapping
    labels = df['label'].unique()
    label_map = {label: idx for idx, label in enumerate(labels)}
    train_set = OxfordPetDataset(df, 'train', label_map, transform)
    val_set = OxfordPetDataset(df, 'val', label_map, transform)
    test_set = OxfordPetDataset(df, 'test', label_map, transform)

    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
    loader_args = dict(batch_size=batch_size, num_workers=4)
    batch_steps = n_train // batch_size
    total_training_steps = epochs * batch_steps

    train_loader = DataLoader(train_set, shuffle=True, **loader_args, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    if scaled_train and old_batch_size is not None:
        learning_rate, beta1, beta2, epsilon = scale_learning_rate(old_batch_size, batch_size, learning_rate, beta1, beta2, epsilon)
    # Initialize a new wandb run and log experiment config parameters; don't forget the run name
    wandb.init(
        project="277 hw1",
        name=run_name,
        config={
            "epoch": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "scheduler": use_scheduler,
            "total_training_steps": total_training_steps,
            "model": "ResNet-18",
            "scaled_train": scaled_train,
            "old_batch_size": old_batch_size,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        }
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(use_scheduler, optimizer, max_lr=learning_rate * 10, total_steps=total_training_steps, pct_start=0.3, final_div_factor=10)  # At 30% of data we reach max_lr

    criterion = nn.CrossEntropyLoss()

    # Initialize timing variables
    data_loading_time = 0.0
    forward_pass_time = 0.0
    backward_pass_time = 0.0
    evaluation_time = 0.0
    total_start_time = time.time()

    # record necessary metrics
    global_step = 0
    seen_examples = 0
    best_val_loss = float('inf')

    # Start PyTorch Profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        # training loop
        for epoch in range(1, epochs + 1):
            model.train()
            with tqdm(total=batch_steps * batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                data_loading_start_time = time.time()  # Initialize data loading timer
                for inputs, labels in train_loader:
                    # Data loading time
                    data_loading_end_time = time.time()
                    data_loading_time += data_loading_end_time - data_loading_start_time

                    with record_function("Data Loading"):
                        inputs, labels = inputs.to(device), labels.to(device)
                    seen_examples += inputs.size(0)

                    # Forward pass
                    start_time = time.time()
                    with record_function("Forward Pass"):
                        outputs = model(inputs)
                    forward_pass_time += time.time() - start_time

                    # Loss calculation and backward pass
                    start_time = time.time()
                    with record_function("Loss and Backward"):
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    backward_pass_time += time.time() - start_time

                    if use_scheduler:
                        scheduler.step()
                        cur_lr = scheduler.get_last_lr()[0]
                    else:
                        cur_lr = optimizer.param_groups[0]['lr']
                    pbar.update(inputs.shape[0])
                    global_step += 1
                    # save necessary metrics in a dictionary; it's recommended to also log seen_examples, which helps you create appropriate figures in Part 3
                    # code here
                    metrics = {
                        'train_loss': loss.item(),
                        # 'seen_examples': seen_examples,
                    }
                    wandb.log(metrics)

                    if global_step % batch_steps == 0:
                        # evaluate on validation set
                        start_eval_time = time.time()
                        with record_function("Evaluation"):
                            val_acc, val_loss, val_loss_step = evaluate(model, val_loader, device)
                        evaluation_time += time.time() - start_eval_time
                        # update metrics from validation results in the dictionary
                        # code here
                        metrics.update({
                            # 'val_loss': val_loss,
                            'val_acc': val_acc,
                        })
                        wandb.log(metrics)

                        if best_val_loss > val_loss:
                            best_val_loss = val_loss
                            os.makedirs(os.path.join(save_dir, f'{run_name}_{rid}'), exist_ok=True)
                            state_dict = model.state_dict()
                            torch.save(state_dict, os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
                            print(f'Checkpoint at step {global_step} saved!')
                        # log metrics to wandb
                        # code here
                        wandb.log({'learning_rate': cur_lr})

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    data_loading_start_time = time.time()  # Start timing data loading for next iteration
            wandb.log({'epoch': epoch})

    # Calculate total time and other time
    total_time = time.time() - total_start_time
    other_time = total_time - (data_loading_time + forward_pass_time + backward_pass_time + evaluation_time)

    # Log timing information to wandb
    wandb.log({
        'data_loading_time': data_loading_time,
        'forward_pass_time': forward_pass_time,
        'backward_pass_time': backward_pass_time,
        'evaluation_time': evaluation_time,
        'other_time': other_time,
        'total_time': total_time,
    })

    # load best checkpoint and evaluate on test set
    print(f'training finished, run testing using best ckpt...')
    state_dict = torch.load(os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
    model.load_state_dict(state_dict)

    start_eval_time = time.time()
    with record_function("Evaluation"):
        test_acc, test_loss, test_loss_step = evaluate(model, test_loader, device)
    evaluation_time += time.time() - start_eval_time

    # Log final test results to wandb
    wandb.log({'final_test_acc': test_acc, 'final_test_loss_step': test_loss_step})
    wandb.summary.update({"final_test_accuracy": test_acc, "final_test_loss": test_loss})
    wandb.finish()
    print(f'test acc: {test_acc}, test loss: {test_loss}')

# ... [rest of your code] ...

