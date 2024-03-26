# training.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from PIL import Image
from utils.dataset import HeadGearDataset
from utils.resnet_18 import resnet18
from utils.resnet_34 import resnet34
from utils.early_stopping import EarlyStopping
from test import test
from sklearn.metrics import f1_score

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import argparse
import os
import subprocess


parser = argparse.ArgumentParser()

parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--w_decay', type=float, default=1e-2, help='weight decay')
parser.add_argument('--g_clip', type=float, default=5.0, help='grad clip threshold')

parser.add_argument('--model_type', type=str, default="18", help="resnet model type")
parser.add_argument('--dropout', type=float, default=0.12, help='dropout rate')
parser.add_argument('--classes', type=int, default=20, help='number of classes')

parser.add_argument('--patience', type=int, default=1, help='patience of early stop')
parser.add_argument('--verbose', type=bool, default=True, help='whether to display messages')

args = parser.parse_args()



def train_one_epoch(model, criterion, optimizer, dataloader, device, grad_clip):
    # TODO: Set the model to train mode
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    all_labels = []
    all_predictions = []

    for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        # TODO: Define Output
        output = model(data)

        # TODO: Define Loss
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)

        _, predicted = torch.max(output.data, 1) # idx (predicted class) of the max val for each data point
        total_predictions += target.size(0) # target.size(0): num of data points in the current batch
        correct_predictions += (predicted == target).sum().item()

        all_labels.extend(target.detach().cpu().numpy().tolist()) # adds the labels of the current batch to the list
        all_predictions.extend(predicted.detach().cpu().numpy().tolist()) # adds the predictions of the current batch to the list

        # TODO: Backpropagate Loss
        loss.backward()
        
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)
        
        # TODO: Update the weights
        optimizer.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = (correct_predictions / total_predictions) * 100.0
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro') # macro-average F1 score: calcuates class-wise F1 scores and averages them

    return epoch_loss, epoch_acc, epoch_f1


def validate(model, criterion, dataloader, device):
    # TODO: Set the model to evaluation mode
    model.eval()

    running_valid_loss = 0.0
    total_valid_predictions = 0.0
    correct_valid_predictions = 0.0
    all_valid_labels = []
    all_valid_predictions = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # TODO: Define Output
            output = model(data)

            # TODO: Define Loss
            loss = criterion(output, target)
            running_valid_loss += loss.item() * data.size(0)

            _, predicted = torch.max(output.data, 1)
            total_valid_predictions += target.size(0)
            correct_valid_predictions += (predicted == target).sum().item()

            all_valid_labels.extend(target.detach().cpu().numpy().tolist())
            all_valid_predictions.extend(predicted.detach().cpu().numpy().tolist())

    valid_loss = running_valid_loss / len(dataloader.dataset)
    valid_acc = (correct_valid_predictions / total_valid_predictions) * 100.0
    valid_f1 = f1_score(all_valid_labels, all_valid_predictions, average='macro')

    return valid_loss, valid_acc, valid_f1

def run_test(model_now):
    print("Running test.py...")
    model_path, test_acc_new, test_f1 = test(model_now)
    
    best_file = 'resnet' + model_now[6:8] + '.txt'
    best_path = os.path.join('./model/best', best_file)

    if os.path.exists(best_path) and os.path.getsize(best_path) > 0:
        with open(best_path, 'r') as f:
            data = f.read()
        test_acc_line = [line for line in data.split('\n') if "test_acc" in line][0]
        test_acc_best = float(test_acc_line.split(": ")[-1].strip())
    else:
        test_acc_best = 0
    
    if test_acc_new > test_acc_best:
        with open(best_path, 'w') as f:
            f.write('model_path: ' + model_path + '\n' + 'test_acc: ' + str(test_acc_new) + '\n' + 'test_f1: ' + str(test_f1))

def training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(64),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
    ])

    # TODO: Load the train dataset using HeadGearDataset class
    train_data = HeadGearDataset("train", transform)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=4)

    # TODO: Load the validation dataset and create a DataLoader for it
    valid_data = HeadGearDataset("valid", transforms.ToTensor())
    valid_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=False, num_workers=4)
    
    # TODO: Define the model, loss function, and optimizer
    if args.model_type == '34':
        model_type = 'resnet34'
        model = resnet34(num_classes=args.classes, dropout_rate=args.dropout)
    else:
        model_type = 'resnet18'
        model = resnet18(num_classes=args.classes, dropout_rate=args.dropout)
    model = model.to(device)
    
    # TODO: Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    
    model_dir = './model'
    early_stopping = EarlyStopping(patience=args.patience, 
                                   verbose=args.verbose,
                                   path=model_dir)
    
    # TODO: Train the model on the training data, and validate it on the validation data
    for epoch in range(args.epochs):
        # TODO: Train the model on the training data
        train_loss, train_acc, train_f1 = train_one_epoch(model, criterion, optimizer, train_loader, device, args.g_clip)
        print(f"Epoch: {epoch+1}/{args.epochs}.. Training Loss: {train_loss:.4f}.. Training Accuracy: {train_acc:.2f}%.. Training F1 Score: {train_f1:.2f}")

        # TODO: Validate the model on the validation data
        valid_loss, valid_acc, valid_f1 = validate(model, criterion, valid_loader, device)
        print(f"Epoch: {epoch+1}/{args.epochs}.. Validation Loss: {valid_loss:.4f}.. Validation Accuracy: {valid_acc:.2f}%.. Validation F1 Score: {valid_f1:.2f}")

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    # TODO: Save the trained model
    model_now = model_type + '_' + datetime.now().strftime("%y%m%d%H")
    save_config(model_now)
    model_save_path = os.path.join(model_dir, model_now + '.pth')
    torch.save(model.state_dict(), model_save_path)

    run_test(model_now)

def save_config(model_now):
    log_path = './model/logs/' + model_now + '.txt'

    with open(log_path, 'w') as f:
        f.write(f'{args}\n\n')

if __name__ == "__main__":
    training()
