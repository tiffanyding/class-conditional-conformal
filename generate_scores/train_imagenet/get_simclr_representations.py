'''
Based on https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/gradient_linear_clf.py
'''


import os
import numpy as np
import argparse
from collections import Counter
import pdb

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from resnet import get_resnet, name_to_params

#class ImagenetValidationDataset(Dataset):
#    def __init__(self, val_path, ground_truth_path): # Modified to take in separate path for ground truth
#        super().__init__()
#        self.val_path = val_path
#        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
#        with open(ground_truth_path) as f:
#            self.labels = [int(l) - 1 for l in f.readlines()]
#
#    def __len__(self):
#        return len(self.labels)
#
#    def __getitem__(self, item):
#        img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
#        return self.transform(img), self.labels[item]



@torch.no_grad()
def run(pth_path, train_or_val, batch_size):
    device = 'cuda'
    path = f'/home/tding/data/imagenet/{train_or_val}' 
#     if train_or_val == 'train':
#         path = '/home/eecs/tiffany_ding/data/imagenet/train' 
#     else:
#         path = 
#         path = '/data/imagenetwhole/ilsvrc2012/val'
#         if not os.path.isdir(path):
#             print('ERROR: TO access ImageNet-val, use the node "ace". Exiting...')
#             exit()
            

    print(f'Loading data from {path}')
    dataset = torchvision.datasets.ImageFolder(path,
                transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    net, _ = get_resnet(*name_to_params(pth_path)) # renamed model --> net

    print('==> loading encoder from checkpoint..')
    net.load_state_dict(torch.load(pth_path)['resnet'])

    print('Number of GPUs available:', torch.cuda.device_count())
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    net = net.to(device)
    net.eval()

    features = [] 
    labels = []
        
    t = tqdm(enumerate(dataloader), total=len(dataloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, targets) in t:
        inputs = inputs.to(device)
        representation = net(inputs)
        features += [representation.cpu()]
        labels += [targets.cpu()]
        # break # TO TEST

    features = torch.cat(features,dim=0)
    labels = torch.cat(labels,dim=0)
    
    save_to = f'/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/simclr_representations/imagenet_{train_or_val}'
    torch.save(features,save_to + '_features.pt')
    torch.save(labels,save_to + '_labels.pt')
    print(f'Saved features and labels to {save_to}_{{features, labels}}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get SimCLR representations')
    parser.add_argument('dataset', type=str, help='which ImageNet dataset to use (train or val)')
    parser.add_argument('--pth_path', type=str, default='r152_3x_sk1.pth',  help='path of the input checkpoint file')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    args = parser.parse_args()
    run(args.pth_path, args.dataset, args.batch_size)


#####


