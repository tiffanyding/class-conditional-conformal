import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, time, copy
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, Subset
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn

from scipy.special import softmax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pdb

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

def calc_mean_std(my_dataset):
    image_data_loader = torch.utils.data.DataLoader(
      my_dataset,
      batch_size=512, 
      shuffle=True, 
      num_workers=2
    )

    X, y = iter(image_data_loader).__next__()

    mean = X.mean(dim=(0,2,3))
    std = X.std(dim=(0,2,3))
    return mean, std

def load_and_process_dataset(dset_fn, target_fn, min_train_instances_class):
    # Standard transform
    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # here do not use transforms.Normalize(mean, std)
    ])
    train_dataset, val_dataset = dset_fn(transform_img)
    
    # Calculate mean and std dev to use when normalizing
    mean, std = calc_mean_std(val_dataset)
    
    # Filter out rare classes 
    if min_train_instances_class > 0:  
        train_targets = target_fn(train_dataset)
        val_targets = target_fn(val_dataset)
        unique_classes, counts = np.unique(train_targets, return_counts=True)
        counts_large_enough = counts >= min_train_instances_class
        final_classes = unique_classes[ counts_large_enough ]
        final_train_idxs = np.where( np.isin(train_targets, final_classes) )[0]
        final_val_idxs = np.where( np.isin(val_targets, final_classes) )[0]

        # Map class labels to consecutive 0,1,2,...
        label_remapping = {}
        idx = 0
        for k in final_classes:
            label_remapping[k] = idx
            idx += 1
        def transform_label(k):
            return label_remapping[k]
        target_transform = transform_label
        
    else:
        final_train_idxs = np.arange(len(train_dataset))
        final_val_idxs = np.arange(len(val_dataset))
        target_transform = None
    
    transform_img = transforms.Compose([transform_img, transforms.Normalize(mean, std)])
    
    train_dataset, val_dataset = dset_fn(transform_img, target_transform=target_transform)

    dataset = torch.utils.data.ConcatDataset([Subset(train_dataset, final_train_idxs), Subset(val_dataset, final_val_idxs)])

    return dataset

def get_dataloaders(config):
    # Load and unpack data
    if config['dataset_name'] == 'iNaturalist':
        def dset_fn(transform_img, target_transform=None):
            train_dataset = datasets.INaturalist(root = '/checkpoints/aa/inaturalist/train', 
                                                 version = '2021_train', 
                                                 download=False, 
                                                 target_type = config['target_type'],
                                                 transform=transform_img,
                                                 target_transform=target_transform) 
            val_dataset = datasets.INaturalist(root = '/checkpoints/aa/inaturalist/val', 
                                               version = '2021_valid', 
                                               download=False, 
                                               target_type = config['target_type'],
                                               transform=transform_img, 
                                               target_transform=target_transform) 
            return train_dataset, val_dataset
        def target_fn(dset):
            return np.array([x[0] for x in dset.index])
    if config['dataset_name'] == 'Places365':
        def dset_fn(transform_img, target_transform=None):
            train_dataset = datasets.Places365(root = '/checkpoints/aa/places/', 
                                               split = 'train-standard', 
                                               download=False, 
                                               transform=transform_img,
                                               target_transform=target_transform) 
            val_dataset = datasets.Places365(root = '/checkpoints/aa/places/', 
                                             split = 'val', 
                                             transform=transform_img, 
                                             download=False,
                                             target_transform=target_transform) 
            return train_dataset, val_dataset
        def target_fn(dset):
            return np.array(dset.targets)

    dataset = load_and_process_dataset(dset_fn, target_fn, config['min_train_instances_class'] )

    assert 0 <= config['frac_val'] <= 1

    generator1 = torch.Generator().manual_seed(0) # For reproducibility
    train, val = torch.utils.data.random_split(dataset, [1-config['frac_val'], config['frac_val']], generator=generator1)
    
    # Create training and validation datasets
    image_datasets = {
        'train' : train,
        'val' : val
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
    save_every_epoch = False # save weights every epoch if accuracy is better than previous best
    
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
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])
                
#                 pdb.set_trace()

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
                
                # Save model weights
                if save_every_epoch: 
                    print(f'Saving epoch {epoch} model')
                    os.makedirs('./.cache', exist_ok=True)
                    torch.save(best_model_wts, './.cache/' + config['model_filename'] + '.pth')
                    with open('./.cache/' + config['model_filename'] + '-config.pkl', 'wb') as f:
                        pickle.dump(config, f)
        
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    torch.save(best_model_wts, './.cache/' + config['model_filename'] + '.pth')
    
    with open('./.cache/' + config['model_filename'] + '-config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    # Save val softmax scores and labels
    val_softmax = np.zeros((len(dataloaders['val'].dataset),config['num_classes']))
    val_labels = np.zeros((len(dataloaders['val'].dataset),), dtype=int)
    j = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['val']):
            inputs = inputs.to(config['device'])
            val_labels[j:j+inputs.shape[0]] = labels.numpy()

            # Get model outputs
            val_softmax[j:j+inputs.shape[0],:] = model(inputs).detach().cpu().numpy()
            j = j + inputs.shape[0]

    # Apply softmax to logits
    val_softmax = softmax(val_softmax, axis=1)
    
    os.makedirs('./.cache', exist_ok=True)
    np.save('./.cache/' + config['model_filename'] + f'-valsoftmax_frac={config["frac_val"]}.npy', val_softmax)
    np.save('./.cache/' + config['model_filename'] + f'-vallabels_frac={config["frac_val"]}.npy', val_labels)
    print('Saved val set softmax scores and labels')

    return model, val_acc_history

def get_model(config):
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model = model.to(config['device'])
    try:
        # ADDED
        assert False
        state_dict = torch.load('./.cache/' + config['model_filename'] + '.pth', map_location=config['device'])
        model.load_state_dict(state_dict)
        model.eval()
        with open('./.cache/' + config['model_filename'] + '-config.pkl', 'rb') as f:
            loaded_config = pickle.load(f)
            
        for setting in ['num_classes', 'batch_size', 'num_epochs', 
                        'frac_val', 'lr', 'dataset_name', 'min_train_instances_class', 'target_type']:
            assert config[setting] == loaded_config[setting] # If the configs aren't equal, retrain

    except:
        model = model.to(config['device'])
        dataloaders = get_dataloaders(config)

        model, val_acc_history = train_model(model, dataloaders, config) 
    return model

def show_img(x):
    x = x.transpose(1,2,0)
    x = (x - x.min())/(x.max() - x.min())
    plt.imshow(x)
    plt.axis('off')
    plt.show()

def postprocess_config(config):
    if config['dataset_name'] == 'Places365':
        config['num_classes'] = 365
    elif config['dataset_name'] == 'iNaturalist' and config['target_type'] == 'full':
        config['num_classes'] = 6414
    elif config['dataset_name'] == 'iNaturalist' and config['target_type'] == 'family':
        config['num_classes'] = 1103
    else:
        raise NotImplementedError
    return config

if __name__ == "__main__":
#     config = {
#             'batch_size' : 128,
#             'lr' : 0.0001,
#             'feature_extract' : False,
#             'num_epochs' : 30,
#             'device' : 'cuda',
#             'frac_val' : 0.1, # For Places 365, this corresponds to >= 500 examples for calibration/val
#             'num_workers' : 4,
#             'dataset_name' : 'Places365',
#             'model_filename' : 'best-places365-model',
#             'target_type': 'full',
#             'min_train_instances_class' : 10 
#     }
    config = {
            'batch_size' : 128,
            'lr' : 0.0001,
            'feature_extract' : False,
            'num_epochs' : 30,
            'device' : 'cuda',
            'frac_val' : 0.5, 
            'num_workers' : 4,
            'dataset_name' : 'iNaturalist',
            'model_filename' : 'best-inaturalist-model',
            'target_type': 'full',
            'min_train_instances_class' : 290 
    }
    config = postprocess_config(config)
    get_model(config)
    
