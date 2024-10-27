import time
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Dataset, Subset


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


def record_time(data_loader, batch_size, num_worker):
    start_time = time.time()
    for _ in tqdm(data_loader, batch_size, desc='load data'):
        pass
    end_time = time.time() - start_time
    print(f'num_worker: {num_worker}, time: {end_time}')
    return end_time


if __name__ == '__main__':
    # Load dataset and csv file
    df = pd.read_csv('oxford_pet_split.csv')
    labels = df['label'].unique()
    label_map = {label: idx for idx, label in enumerate(labels)}
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])  # Use ImageNet
    dataset = datasets.OxfordIIITPet(root='data', download=True, transform=transform)
    # Image-label mapping
    train_set = OxfordPetDataset(df, 'train', label_map, transform)
    n_train = len(train_set)
    # hyperparameters
    epochs = 10
    batch_size = 32
    loading_time = []
    for num_worker in tqdm(range(1, 11), desc='num_worker'):
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_worker, pin_memory=True, shuffle=True)
        time_taken = record_time(train_loader, batch_size, num_worker)
        loading_time.append(time_taken)

    # Plot result
    plt.plot(range(1, 11), loading_time, marker='o')
    plt.xlabel('num_worker')
    plt.ylabel('time')
    plt.title('Loading time vs num_worker')
    plt.grid(True)
    plt.show()











