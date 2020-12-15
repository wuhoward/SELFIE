import torch
import os
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import numpy as np
from preact_resnet import *

MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
STD = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

FEATURE_SIZE = 1024 # Output of ResNet Model
PATCH_SIZE = 8
use_gpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

def get_dataloader(batch_size):
    data_transform = transforms.ToTensor()
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
    trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='.', train=False, download=True, transform=test_transform)
    
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return dataloaders

class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_pos):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=32, dim_feedforward=640, dropout=0.1, activation='gelu')
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, self.encoder_norm)
        self.pos_embedding = nn.Embedding(num_pos, hidden_dim)

    def forward(self, x, pos):
        v = self.pos_embedding(pos)
        u = self.encoder(x)[0]
        num_batch = u.shape[0]
        num_target = pos.shape[0] // num_batch
        u = u.repeat(1, num_target).reshape(num_batch*num_target, -1)
        v += u
        return v.reshape(num_batch, num_target, -1)

def train_model(model, encoder, optimizer, dataloaders, args):
    warmup_step = min(math.ceil(len(dataloaders['train'].dataset) / args.batch_size), args.warm_up) + 1 
    if warmup_step > 0:
        warmup_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=warmup_step * 2, pct_start=0.5, anneal_strategy='linear')
    else:
        warmup_scheduler = None
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    current_step = 0
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
                encoder.train()
            else:
                model.eval()   # Set model to evaluate mode
                encoder.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, _ in dataloaders[phase]:
                inputs = inputs.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    num_patch = (inputs.shape[2] // PATCH_SIZE) ** 2

                    # [N*num_patch, 3, 8, 8]
                    patches = inputs.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE).permute(
                        0, 2, 3, 1, 4, 5).reshape(-1, 3, 8, 8)

                    num_target = int(num_patch * 0.25)

                    # Feed all the patches into the feature extractor
                    hidden = model(patches).reshape(len(inputs), num_patch, -1)
                    hidden_dim = hidden.shape[2]

                    # use the largest randon scores to select the target patches
                    random_score = torch.rand(len(inputs), num_patch)

                    # Separate context patches from the target patches
                    # [N, num_patch]
                    target_index = torch.topk(random_score, k=num_target, dim=1).indices
                    context_index = torch.topk(random_score, k=num_patch-num_target, largest=False, dim=1).indices
                    context_patch = hidden[torch.arange(len(inputs)).unsqueeze(1), context_index]
                    target_patch = hidden[torch.arange(len(inputs)).unsqueeze(1), target_index]

                    u_0 = torch.zeros([len(inputs), FEATURE_SIZE]).unsqueeze(0).to(device)
                    u = encoder(torch.cat((u_0, context_patch.permute(1, 0 ,2))), target_index.reshape(-1).to(device))
                    outputs = torch.bmm(u, target_patch.permute(0, 2, 1)).reshape(-1, num_target)
                    targets = torch.arange(num_target).repeat(len(inputs)).to(device)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if epoch == 0 and current_step < warmup_step - 1:
                            warmup_scheduler.step()
                            current_step += 1
                            
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets)

            if phase == 'train':
                if warmup_step == 0 or warmup_step > 0 and epoch != 0: 
                    cosine_scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * num_target)
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
                os.makedirs('checkpoint/selfie', exist_ok=True)
                torch.save(state, './checkpoint/selfie/ckpt_{}.pth'.format(epoch))
                best_acc = epoch_acc
    
    os.makedirs('logdir/selfie', exist_ok=True)
    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        np.save('logdir/selfie/' + name, perf_dict[name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--val-cycle', default=1, type=int, help='perform validation every ? epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--total-epoch', default=200, type=int, help='total epochs to train')
    parser.add_argument('--warm-up', default=1000, type=int, help='number of warmup steps')
    parser.add_argument('--resume-selfie', default=None, type=str, help='resume from selfie checkpoint')
    
    args = parser.parse_args()

    dataloaders = get_dataloader(args.batch_size)
    model = PreActResNet50()
    model.layer4 = nn.Identity()
    model.linear = nn.Identity()
    model = model.to(device)
    encoder = Encoder(FEATURE_SIZE, 3, 16)
    encoder = encoder.to(device)
        
    optimizer = optim.SGD(list(model.parameters()) + list(encoder.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    train_model(model, encoder, optimizer, dataloaders, args)
