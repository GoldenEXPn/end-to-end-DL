import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

### Data Loading

class OxfordPetsDataset(Dataset):
    def __init__(self, label_df, label_map, img_dir, partition):
        self.images = list(label_df[label_df['split'] == partition]['image_name'])
        self.labels = list(label_df[label_df['split'] == partition]['label'])
        assert len(self.images) == len(self.labels) and len(self.images) > 0, 'Invalid dataset'
        self.label_map = label_map
        self.root_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.labels[idx]]
        image = self.transform(image)
        return image, label


def create_online_dataset(label_file, img_dir, num_split=5):
    label_df = pd.read_csv(label_file)
    label_list = sorted(label_df['label'].unique())
    num_class_per_split = len(label_list) // num_split
    label_list = label_list[:num_split * num_class_per_split]

    label_map = {label: i for i, label in enumerate(label_list)}
    print(f"number of classes: {len(label_list)}")
    train_data_stream = []
    test_data_stream = []
    for i in range(num_split):
        selected_label = label_list[i * num_class_per_split:(i + 1) * num_class_per_split]
        subset_df = label_df[label_df['label'].isin(selected_label)]
        train_data_stream.append(OxfordPetsDataset(subset_df, label_map, img_dir, partition='train'))
        test_data_stream.append(OxfordPetsDataset(subset_df, label_map, img_dir, partition='test'))
    return train_data_stream, test_data_stream, len(label_list)


#### Model Training

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()
        prediction = model(batch_data[0].to(device))
        loss = criterion(prediction, batch_data[1].to(device))
        loss.backward()
        optimizer.step()
    return


def evaluate(model, test_loader, device):
    model.eval()
    samples = 0
    correct = 0
    for batch_data in test_loader:
        prediction = model(batch_data[0].to(device))
        labels = batch_data[1].to(device)
        correct += torch.sum(torch.argmax(prediction, dim=1) == labels).item()
        samples += len(labels)
    accuracy = correct / samples
    return accuracy


def online_training(model, train_stream, test_stream, device, batch_size=32):
    learned_set = []
    for i, (train_set, test_set) in enumerate(zip(train_stream, test_stream)):
        print(f"Training on split {i}")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        learned_set.append(test_set)
        for test_idx, test_set in enumerate(learned_set):
            test_loader = DataLoader(test_set, batch_size=batch_size)
            accuracy = evaluate(model, test_loader, device)
            print(f"\t Accuracy on split {test_idx}: {accuracy}")
    return


#### SLDA Training

def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat


def train_one_epoch_slda(slda, feature_extractor, train_loader, device):
    for batch_data in train_loader:
        batch_x_feat = feature_extractor(batch_data[0].to(device))
        batch_x_feat = pool_feat(batch_x_feat)
        batch_y = batch_data[1]
        for x, y in zip(batch_x_feat, batch_y):
            slda.fit(x, y)
    return


def evaluate_slda(slda, feature_extractor, test_loader, device):
    samples = 0
    correct = 0
    for batch_data in test_loader:
        batch_x_feat = feature_extractor(batch_data[0].to(device))
        batch_x_feat = pool_feat(batch_x_feat)
        prediction = slda.predict(batch_x_feat).to(device)
        labels = batch_data[1].to(device)
        correct += torch.sum(torch.argmax(prediction, dim=1) == labels).item()
        samples += len(labels)
    accuracy = correct / samples
    return accuracy


def online_training_slda(slda, feature_extractor, train_stream, test_stream, device, batch_size=32):
    learned_set = []
    for i, (train_set, test_set) in enumerate(zip(train_stream, test_stream)):
        print(f"Training on split {i}")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        train_one_epoch_slda(slda, feature_extractor, train_loader, device)
        learned_set.append(test_set)
        for test_idx, test_set in enumerate(learned_set):
            test_loader = DataLoader(test_set, batch_size=batch_size)
            accuracy = evaluate_slda(slda, feature_extractor, test_loader, device)
            print(f"\t Accuracy on split {test_idx}: {accuracy}")



### model wrapper

def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))

class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, x):
        self.model(x)
        output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals