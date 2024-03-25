# test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import HeadGearDataset
from utils.resnet_18 import resnet18
from utils.resnet_34 import resnet34
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--model_type', type=str, default='18', help='resnet model type')
parser.add_argument('--classes', type=int, default=20, help='number of classes')
args = parser.parse_args()

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # TODO: Load test dataset using the HeadGearDataset class
    test_data = HeadGearDataset("test", transform=transform)

    # TODO: Define data loader for test set
    test_loader = DataLoader(test_data, args.batch, shuffle=False)

    # TODO: Define the model and load the trained weights
    if args.model_type == '34':
        model_type = 'resnet34'
        model = resnet34(num_classes=args.classes)
    else:
        model_type = 'resnet18'
        model = resnet18(num_classes=args.classes)
    
    model_dir = '/NasData/home/kmg/mlcl/MLCL_2023/3_headgear/classification_new/model/'
    files = os.listdir(model_dir)
    matching_files = [file for file in files if file.startswith(model_type)]
    latest_timestamp = max([int(file.split('_')[1][:8]) for file in matching_files])
    latest_model = [file for file in matching_files if file.startswith(model_type + '_' + str(latest_timestamp))]

    # TODO: Load the model
    model.load_state_dict(torch.load(model_dir + latest_model[0]))
    model.to(device)
    
    # TODO: Set the model to test mode
    model.eval()

    total_predictions = 0.0
    correct_predictions = 0.0
    all_labels = []
    all_predictions = []

    # TODO: Stop tracking gradients
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            
            # TODO: Define Output
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predicted.detach().cpu().numpy().tolist())

    test_acc = (correct_predictions / total_predictions) * 100.0
    test_f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Test Accuracy: {test_acc:.2f}%.. Test F1 Score: {test_f1:.2f}")

if __name__ == "__main__":
    test()
