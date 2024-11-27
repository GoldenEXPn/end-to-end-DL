import torch
from torch import nn
import argparse
from torchvision.models import resnet18
from slda import StreamingLDA
from utils import create_online_dataset, online_training, online_training_slda, ModelWrapper


def get_args():
    parser = argparse.ArgumentParser(description='E2EDL training script')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--label_file', help='path to the label file', default='./oxford_pet_split.csv')
    parser.add_argument('--img_dir', help='path to the image directory')
    parser.add_argument('--use_slda', action='store_true', help='use streaming lda')
    parser.add_argument('--num_split', type=int, default=5, help='number of chunks to split the dataset into')

    # IMPORTANT: if you are copying this script to notebook, replace 'return parser.parse_args()' with 'args = parser.parse_args("")'
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_stream, test_data_stream, num_class = create_online_dataset(args.label_file, args.img_dir, num_split=args.num_split)

    if not args.use_slda:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_class)
        model.to(device)
        online_training(model, train_data_stream, test_data_stream, device, batch_size=args.batch_size)
    else:
        slda = StreamingLDA(512, num_class, device=device)
        feature_extractor = resnet18(pretrained=True)
        feature_extraction_wrapper = ModelWrapper(feature_extractor.to(device), ['layer4.1'], return_single=True).eval()
        online_training_slda(slda,feature_extraction_wrapper, train_data_stream, test_data_stream, device, batch_size=args.batch_size)
