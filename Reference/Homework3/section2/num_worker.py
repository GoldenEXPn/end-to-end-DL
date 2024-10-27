import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # Progress tracking

class OxfordPetsDataset(Dataset):
    def __init__(self, dataframe, split, label_map, transform=None):
        self.dataframe = dataframe[dataframe['split'] == split]
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Debug: Check if image path is correct
        img_name = self.dataframe.iloc[idx]['image_name']
        img_path = f"Homework1/data/images/{img_name}"
        label = self.dataframe.iloc[idx]['label']

        try:
            image = Image.open(img_path).convert('RGB')  # Ensure image loads correctly
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
            raise

        if self.transform:
            image = self.transform(image)

        return image, self.label_map[label]

def benchmark_num_workers(data_loader, num_batches):
    # In this case, I think using perf_counter rather than process_counter is preferred for benchmark testing.
    start_time = time.perf_counter()
    max_batches = min(num_batches, len(data_loader))

    for images, labels in tqdm(data_loader, total=max_batches, desc="Benchmarking"):
        pass  

    elapsed_time = time.perf_counter() - start_time
    print(f"Time for {max_batches} batches: {elapsed_time:.2f} seconds")
    return elapsed_time

def main():
    df = pd.read_csv('Homework1/problem_1/oxford_pet_split.csv')
    df['label_code'], unique_labels = pd.factorize(df['label'])
    label_map = dict(zip(unique_labels, range(len(unique_labels))))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = OxfordPetsDataset(dataframe=df, split='train', label_map=label_map, transform=transform)

    num_workers_list = range(1, 11)
    loading_times = []

    # Track outer loop progress
    for num_workers in tqdm(num_workers_list, desc="Testing num_workers"):
        loader = DataLoader(train_set, batch_size=8, num_workers=num_workers, shuffle=False)

        # Benchmark with 10 batches and log time
        time_taken = benchmark_num_workers(loader, num_batches=10)
        loading_times.append(time_taken)

    # Plot the results
    plt.plot(num_workers_list, loading_times, marker='o')
    plt.xlabel('Number of workers')
    plt.ylabel('Loading time (seconds)')
    plt.title('Effect of num_workers on DataLoader Performance')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
