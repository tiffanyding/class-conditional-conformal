import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, time, copy
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pdb

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data():
    # Load and unpack data
    orig_train_data = unpickle('../data/cifar-100-python/train')
    orig_test_data = unpickle('../data/cifar-100-python/test')
    tr_imgs = orig_train_data[b'data'].astype(np.float32)
    te_imgs = orig_test_data[b'data'].astype(np.float32)
    tr_labels = torch.tensor(np.array(orig_train_data[b'fine_labels']).astype(int))
    te_labels = torch.tensor(np.array(orig_test_data[b'fine_labels']).astype(int))

    # Fuse train and val sets
    imgs = np.concatenate([tr_imgs, te_imgs], axis=0)
    labels = np.concatenate([tr_labels, te_labels], axis=0)

    # Reshape and normalize images to mean 0 std 1
    imgs = imgs.reshape(imgs.shape[0], 3, 32, 32)
    total_pixels_per_channel = imgs.shape[0] * imgs.shape[2] * imgs.shape[3] 
    means = imgs.sum(axis=2).sum(axis=2).sum(axis=0) / total_pixels_per_channel
    stds = np.sqrt(((imgs - means[None,:,None,None])**2).sum(axis=2).sum(axis=2).sum(axis=0)/total_pixels_per_channel)
    imgs = (imgs - means[None,:,None,None])/stds[None,:,None,None]
    return imgs, labels

def get_dataloaders(config, frac_val=0.1):
    assert 0 <= frac_val <= 1

    imgs, labels = get_data()

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 224 is due to Imagenet input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    }

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=frac_val, random_state=0)

    # Create training and validation datasets
    image_datasets = {
        'train' : TensorDataset(torch.tensor(train_imgs).float(), torch.tensor(train_labels).long()),
        'val' : TensorDataset(torch.tensor(val_imgs).float(), torch.tensor(val_labels).long()) 
                     }
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']) for x in ['train', 'val']}

    return dataloaders_dict

def train_model(model, dataloaders, config):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    set_parameter_requires_grad(model, config['feature_extract'])

    params_to_update = model.parameters()
    print("Params to learn:")
    if config['feature_extract']:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # The above prints show which layers are being optimized

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params_to_update, lr=config['lr'])

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        print('Epoch {}/{}'.format(epoch, config['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    os.makedirs('./.cache', exist_ok=True)
    torch.save(best_model_wts, './.cache/' + config['model_filename'] + '.pth')
    np.save('./.cache/' + config['model_filename'] + f'-valdata_frac={config["frac_val"]}.npy', dataloaders['val'].dataset.tensors[0].numpy())
    np.save('./.cache/' + config['model_filename'] + f'-vallabels_frac={config["frac_val"]}.npy', dataloaders['val'].dataset.tensors[1].numpy())
    with open('./.cache/' + config['model_filename'] + '-config.pkl', 'wb') as f:
        pickle.dump(config, f)

    return model, val_acc_history

def get_model(config):
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model = model.to(config['device'])
    try:
        state_dict = torch.load('./.cache/' + config['model_filename'] + '.pth', map_location=config['device'])
        model.load_state_dict(state_dict)
        model.eval()
        with open('./.cache/' + config['model_filename'] + '-config.pkl', 'rb') as f:
            loaded_config = pickle.load(f)
        assert config['num_classes'] == loaded_config['num_classes'] # If the configs aren't equal, retrain
        assert config['batch_size'] == loaded_config['batch_size'] # If the configs aren't equal, retrain
        assert config['num_epochs'] == loaded_config['num_epochs'] # If the configs aren't equal, retrain
        assert config['frac_val'] == loaded_config['frac_val'] # If the configs aren't equal, retrain
        assert config['lr'] == loaded_config['lr'] # If the configs aren't equal, retrain
    except:
        model = model.to(config['device'])
        dataloaders = get_dataloaders(config, frac_val = config['frac_val'])

        model, val_acc_history = train_model(model, dataloaders, config) 
    return model

def show_img(x):
    x = x.transpose(1,2,0)
    x = (x - x.min())/(x.max() - x.min())
    plt.imshow(x)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    config = {
            'num_classes' : 100,
            'batch_size' : 128,
            'lr' : 0.0001,
            'feature_extract' : False,
            'num_epochs' : 30,
            'device' : 'cuda',
            'frac_val' : 0.7, # CHANGED
            'model_filename' : 'best-cifar100-model-fracval=0.7', # CHANGED
            'num_workers' : 4,
    }
    get_model(config)
