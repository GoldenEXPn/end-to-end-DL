import os
import time
import torch
import wandb
import random
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from utils import set_seed
from torch import nn, optim, autocast, GradScaler
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset


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
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size; modify this to fit your GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_scheduler', action='store_true', help='use lr scheduler')
    return parser.parse_args()


def evaluate(model, data_loader, device):
    """
    :param model: instance of model
    :param data_loader: instance of data loader
    :param device: cpu or cuda
    :return: accuracy, cross entropy loss (sum)
    """
    model.eval()
    num_instances, correct_predictions = 0, 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            num_instances += labels.size(0)
    avg_loss = total_loss / num_instances
    accuracy = correct_predictions / num_instances
    model.eval()
    return accuracy, avg_loss, loss.item()


class OxfordPetDataset(Dataset):
    def __init__(self, csv_file, split, label_map, transform=None):
        self.csv_file = csv_file[csv_file['split'] == split]
        self.label_map = label_map  # Filter by train or test
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = self.csv_file.iloc[idx]['image_name']
        img_path = f"data/oxford-iiit-pet/images/{img_name}"
        label = self.csv_file.iloc[idx]['label']
        label = self.label_map[label]

        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

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
        use_amp=False
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
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Code profiling variables
    data_loading_time, forward_time, backward_time, eval_time, total_time = 0.0, 0.0, 0.0, 0.0, 0.0
    total_start_time = time.time()

    # record necessary metrics
    global_step = 0
    seen_examples = 0
    best_val_loss = float('inf')

    # training loop wrapped by profiler
    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=batch_steps * batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            data_loading_start_time = time.time()
            for inputs, labels in train_loader:
                # Data loading
                inputs, labels = inputs.to(device), labels.to(device)
                seen_examples += inputs.size(0)

                # Forward pass with amp
                if use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                # Backward pass
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Logging
                cur_lr = optimizer.param_groups[0]['lr']
                pbar.update(inputs.shape[0])
                global_step += 1
                metrics = {
                    'train_loss': loss.item(),
                    # 'seen_examples': seen_examples,
                }
                wandb.log(metrics)

                if global_step % batch_steps == 0:
                    # Evaluation
                    val_acc, val_loss, val_loss_step = evaluate(model, val_loader, device)
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
                    wandb.log({'learning_rate': cur_lr})
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                data_loading_start_time = time.time()
        wandb.log({'epoch': epoch})

    # Log Code profiling
    total_time = time.time() - total_start_time
    other_time = total_time - (data_loading_time + forward_time + backward_time + eval_time)
    wandb.log({'total_time': total_time})

    # load the best checkpoint and evaluate on test set
    print(f'training finished, run testing using best ckpt...')
    state_dict = torch.load(os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
    model.load_state_dict(state_dict)
    test_acc, test_loss, test_loss_step = evaluate(model, test_loader, device)

    # log test results to wandb
    # code here
    wandb.log({'final_test_acc': test_acc, 'final_test_loss_step': test_loss_step})
    wandb.summary.update({"final_test_accuracy": test_acc, "final_test_loss": test_loss})
    wandb.finish()
    print(f'test acc: {test_acc}, test loss: {test_loss}')


if __name__ == '__main__':
    # Finetune Pretrained
    rid = random.randint(0, 1000000)
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=True)
    num_classes = 37
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    train_model(
        run_name=args.run_name,
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        use_scheduler=args.use_scheduler,
        rid=rid
    )