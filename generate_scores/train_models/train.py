import argparse

from torchvision_dataset_utils import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('dataset', type=str, choices=['Places365', 'iNaturalist'],
                        help='Name of the dataset to train model on')
    parser.add_argument('frac_val', type=float,
                        help='Fraction of data to reserve for validation')
    parser.add_argument('--min_train_instances', type=int, default=0,
                    help='Classes with fewer than this many classes in the published train dataset will be filtered out')
    parser.add_argument('--num_epochs', type=int, default=30,
                    help='Number of epochs to train for')
    parser.add_argument('--target_type', type=str, default='full',
                    help="Only used when dataset==iNaturalist. Options are ['full', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'] ")
    
    args = parser.parse_args()
    
    config = {
            'batch_size' : 128,
            'lr' : 0.0001,
            'feature_extract' : False,
            'num_epochs' : args.num_epochs,
            'device' : 'cuda',
            'frac_val' : args.frac_val,
            'num_workers' : 4,
            'dataset_name' : args.dataset,
            'model_filename' : f'best-{args.dataset}-model',
            'target_type': args.target_type,
            'min_train_instances_class' : args.min_train_instances
    }
    config = postprocess_config(config)
    get_model(config)