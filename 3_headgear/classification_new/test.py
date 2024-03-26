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
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=64, help='batch size')
#parser.add_argument('--model_type', type=str, default='18', help='resnet model type')
parser.add_argument('--classes', type=int, default=20, help='number of classes')
parser.add_argument('--model', type=str, default='', help='name of model file')
args = parser.parse_args()

def get_model(model_name):
    # get the specified model 
    if os.path.exists('./model/' + model_name + '.pth'):
        return './model/' + model_name + '.pth'

    # get the best model of certain resnet structure (18, 34)
    if model_name[6:8] == '34':
        model_type = '34'
    else:
        model_type = '18'

    best_path = './model/best/resnet' + model_type + '.txt'
    if os.path.exists(best_path) and os.path.getsize(best_path) > 0:
        with open(best_path, 'r') as f:
            data = f.read()
        model_path_line = [line for line in data.split('\n') if 'model_path' in line]
        if model_path_line:
            model_path = model_path_line[0].split(": ")[-1].strip()
            return model_path

    # get the latest model of certain resnet structure (18, 34)
    files = os.listdir('./model/')
    matching_files = [file for file in files if file.startswith(model_type)]
    latest_timestamp = max([int(file.split('_')[1][:8]) for file in matching_files])
    latest_model = [file for file in matching_files if file.startswith(model_type + '_' + str(latest_timestamp))]
    if len(latest_model) > 0:
        return latest_model[0]
    
    return None

def test(model_name=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # TODO: Load test dataset using the HeadGearDataset class
    test_data = HeadGearDataset("test", transform=transform)

    # TODO: Define data loader for test set
    test_loader = DataLoader(test_data, args.batch, shuffle=False)

    # TODO: Define the model and load the trained weights
    if len(args.model) > 0:
        model_name = args.model

    if model_name[6:8] == '34':
        model = resnet34(num_classes=args.classes)
    else:
        model = resnet18(num_classes=args.classes)

    # TODO: Load the model
    model_path = get_model(model_name)
    if model_path is None:
        print("Failed to find a model to test.")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print("Failed to load the model:", e)
        sys.exit(1)

    model_path = os.path.abspath(model_path)
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

    print(f"Model Path: {model_path}")
    print(f"Test Accuracy: {test_acc:.2f}%.. Test F1 Score: {test_f1:.2f}")

    return model_path, test_acc, test_f1

if __name__ == "__main__":
    test()
