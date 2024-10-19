import os
import clip
import torch
import wandb
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ImageEncoder, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model
        self.fc = nn.Linear(self.clip_model.visual.output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x)
        x = self.relu(self.fc(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(TextEncoder, self).__init__()
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2').to(device)  # Move all layers to GPU
        self.fc = nn.Linear(384, output_dim)  # sBERT embedding size is 384
        self.relu = nn.ReLU()

    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.sbert.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.to(device)
        x = self.relu(self.fc(embeddings))
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes=30, embedding_dim=128):
        super(CombinedModel, self).__init__()
        self.image_encoder = ImageEncoder(output_dim=embedding_dim)
        self.text_encoder = TextEncoder(output_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim*2, num_classes)
        )

    def forward(self, images, texts):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(texts)
        x = torch.cat((image_embeddings, text_embeddings), dim=1)
        x = self.classifier(x)
        return x


class VQADataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.loc[idx, 'question']
        image_id = self.df.loc[idx, 'image_id']
        label = torch.tensor(int(self.df.loc[idx, 'label']), dtype=torch.long)
        # Load and preprocess the image
        image_path = os.path.join(self.image_dir, f'{image_id}.png')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, question, label


def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, questions, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            texts = list(questions)  # Move to CPU

            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return total_loss/total, correct/total

if __name__ == '__main__':
    # hyperparams
    num_epochs, lr, batch_size = 20, 5e-3, 64

    # Load data
    train_data = pd.read_csv('new_data_train.csv')
    val_data = pd.read_csv('new_data_val.csv')
    test_data = pd.read_csv('new_data_test.csv')

    # Encode answers to labels
    le = LabelEncoder()
    le.fit(pd.concat([train_data['answer'], val_data['answer'], test_data['answer']]))
    train_data['label'] = le.transform(train_data['answer'])
    val_data['label'] = le.transform(val_data['answer'])
    test_data['label'] = le.transform(test_data['answer'])

    # Transform images
    img_dir = '../data/images'
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataloaders
    train_dataset = VQADataset(train_data, img_dir, image_transforms)
    val_dataset = VQADataset(val_data, img_dir, image_transforms)
    test_dataset = VQADataset(test_data, img_dir, image_transforms)
    num_workers = 4  # Speed up data transfer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Train the model
    wandb.init(
        project='277_hw2',
        name=f'ResNet_SBERT_lr={lr}_batch_size={batch_size}_epochs={num_epochs}',
        config={
            "epoch": num_epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "model": ["ResNet-50", "all-MiniLM-L6-v2"],
        }
    )
    model = CombinedModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
    for n in range(num_epochs):
        # Train
        model.train()
        total_loss = 0.0
        for images, questions, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            texts = list(questions)  # Text processed within the model

            optimizer.zero_grad()
            outputs = model(images, texts)
            # print(f'outputs.dtype: {outputs.dtype}, labels.dtype: {labels.dtype}')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({'train_loss_step': loss.item()})
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_loss, val_accuracy = compute_accuracy(model, val_loader)
        scheduler.step(val_loss)
        wandb.log({'epoch': n, 'train_loss': avg_train_loss, 'val_accuracy': val_accuracy, 'val_loss': val_loss})
        print(f'Epoch {n+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}')

    # Test
    test_loss, test_accuracy = compute_accuracy(model, test_loader)
    wandb.log({'test_accuracy': test_accuracy, 'test_loss': test_loss})
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    torch.save(model.state_dict(), 'ResNet_SBERT.pth')








