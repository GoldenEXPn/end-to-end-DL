import os
import torch
import wandb
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.feature_extrator = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last layer
        self.fc = nn.Linear(self.resnet.fc.in_features,  output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extrator(x)
            x = x.View(x.size(0), -1)  # Make 1-d
        x = self.relu(self.fc(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(TextEncoder, self).__init__()
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.fc = nn.Linear(384, output_dim)
        self.relu = nn.ReLU()

    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.to(device)
        x = self.relu(self.fc(embeddings))
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes=30, embedding_dim=128):
        super(CombinedModel, self).__init__()
        self.image_encoder = ImageEncoder(output_dim=embedding_dim)
        self.text_encoder = TextEncoder(output_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim*2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, texts):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(texts)
        x = torch.cat((image_embeddings, text_embeddings), dim=1)
        x = self.sigmoid(self.classifier(x))
        return x


class VQADataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.loc


if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv('new_data_train.csv')
    val_data = pd.read_csv('new_data_val.csv')
    test_data = pd.read_csv('new_data_test.csv')



































