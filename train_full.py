import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils, models
from torch.utils.data import Dataset, DataLoader
from preact_resnet import *
import argparse


use_gpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
STD = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
np.random.seed(0)

def get_dataloader(batch_size, train_percent=1):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataloaders = {}
    trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='.', train=False, download=True, transform=test_transform)
    trainidx = np.random.choice(len(trainset), int(len(trainset) * train_percent), replace=False)
    trainset = torch.utils.data.Subset(trainset, trainidx)
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloaders['val']  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return dataloaders


def train_model(model, optimizer, dataloaders, args):
    os.makedirs('checkpoint/baseline', exist_ok=True)
    os.makedirs('checkpoint/finetune', exist_ok=True)
    os.makedirs('logdir/baseline', exist_ok=True) 
    os.makedirs('logdir/finetune', exist_ok=True) 

    warmup_step = min(math.ceil(len(dataloaders['train'].dataset) / args.batch_size), args.warm_up) + 1 
    if warmup_step > 0:
        warmup_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=warmup_step * 2, pct_start=0.5, anneal_strategy='linear')
    else:
        warmup_scheduler = None
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    current_step = 0
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['net'])
        #best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if args.resume_selfie:
        print('==> Resuming from selfie checkpoint..')
        checkpoint = torch.load(args.resume_selfie, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
    perf_dict = {}
    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        perf_dict[name] = []

    for epoch in range(args.total_epoch):
        print('Epoch {}/{}'.format(epoch, args.total_epoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        if (epoch+1) % args.val_cycle == 0:
            phases = ['train', 'val']
        else:
            phases = ['train']

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if epoch == 0 and current_step < warmup_step - 1:
                            warmup_scheduler.step()
                            current_step += 1
                            
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                if warmup_step == 0 or warmup_step > 0 and epoch != 0: 
                    cosine_scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            perf_dict[phase + '_loss'].append(epoch_loss)
            perf_dict[phase + '_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                print('==> Saving the new best model with val acc {}'.format(epoch_acc))
                state = {
                    'net': model.state_dict(),
                    'acc': epoch_acc,
                    'epoch': epoch,
                }
                if args.resume_selfie:
                    torch.save(state, './checkpoint/finetune/ckpt_{}.pth'.format(epoch))
                else:
                    torch.save(state, './checkpoint/baseline/ckpt_{}.pth'.format(epoch))
                best_acc = epoch_acc

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        if args.resume_selfie:
            np.save('logdir/finetune/' + name, perf_dict[name])
        else:
            np.save('logdir/baseline', perf_dict[name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--val-cycle', default=1, type=int, help='perform validation every ? epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--total-epoch', default=200, type=int, help='total epochs to train')
    parser.add_argument('--train-percent', default=1, type=float, help='percentage of training data')
    parser.add_argument('--warm-up', default=1000, type=int, help='number of warmup steps')
    parser.add_argument('--resume', default=None, type=str, help='resume from selfie checkpoint')
    parser.add_argument('--resume-selfie', default=None, type=str, help='resume from selfie checkpoint')
    
    args = parser.parse_args()
    print(args)

    dataloaders = get_dataloader(args.batch_size, args.train_percent)
    model = PreActResNet50()
    model = model.to(device)
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    train_model(model, optimizer, dataloaders, args)
