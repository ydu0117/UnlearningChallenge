import os
import subprocess
from pathlib import Path
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
def load_example(df_row):
    image = torchvision.io.read_image(df_row['image_path'])
    result = {
        'image': image,
        'image_id': df_row['image_id'],
        'age_group': df_row['age_group'],
        'age': df_row['age'],
        'person_id': df_row['person_id']
    }
    # Update counter
    load_example.counter += 1

    # Compute percentage
    percentage = 100 * (load_example.counter / load_example.total)

    # Print percentage
    print('Loading progress: {:.2f}%'.format(percentage), end='\r')
    # Check for 3 in first dimension (3 color channels)
    if image.size(0) != 3:
        result = None

    return result


class HiddenDataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, split='train'):
        super().__init__()
        self.examples = []

        df = pd.read_csv(f'/kaggle/input/neurips-2023-machine-unlearning/{split}.csv')
        df['image_path'] = df['image_id'].apply(
            lambda x: os.path.join('/kaggle/input/neurips-2023-machine-unlearning/', 'images', x.split('-')[0], x.split('-')[1] + '.png'))
        df = df.sort_values(by='image_path')
        df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
        if len(self.examples) == 0:
            raise ValueError('No examples.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example['image']
        image = image.to(torch.float32)
        example['image'] = image
        return example

class OurDataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, root_path, split='train'):
        super().__init__()
        self.examples = []
        self.root_path = root_path

        self.transform = T.Compose([
            T.ToPILImage(),
            # T.Grayscale(num_output_channels=3),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        df = pd.read_csv(os.path.join('./data', f'{split}.csv'))
        df['image_path'] = df['image_id'].apply(
            lambda x: os.path.join(root_path, x.replace('\\', '/')))
        df = df.sort_values(by='image_path')

        # Initialize counter and total attributes
        load_example.counter = 0
        load_example.total = df.shape[0]
        df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
        self.examples = [ex for ex in self.examples if ex is not None]

        if len(self.examples) == 0:
            raise ValueError('No examples.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example['image']
        # Resize to 256x256 and overwrite the image
        image = self.transform(image)
        image = image.to(torch.float32)
        example['image'] = image
        return example


def get_dataset(batch_size):
    '''Get the dataset.'''
    retain_ds = HiddenDataset(split='retain')
    forget_ds = HiddenDataset(split='forget')
    val_ds = HiddenDataset(split='validation')

    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return retain_loader, forget_loader, validation_loader


def build_dataset(root_path, batch_size, shuffle, mode):

    train_loader = None
    validation_loader = None
    test_loader = None

    if mode == 'train':
        '''Get the dataset.'''
        training_set = OurDataset(split='train', root_path=root_path)
        val_set = OurDataset(split='val', root_path=root_path)

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)

        print('')
        print('Trainset: %d' % len(training_set))
        print('Valset: %d' % len(val_set))

    elif mode == 'retain':
        '''Get the dataset.'''
        training_set = OurDataset(split='retain', root_path=root_path)
        val_set = OurDataset(split='val', root_path=root_path)
        # forget_set = OurDataset(split='forget', root_path=root_path)

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)

        print('')
        print('Retainset: %d' % len(training_set))
        print('Valset: %d' % len(val_set))

    elif mode == 'test':
        '''Get the dataset.'''
        test_set = OurDataset(split='test', root_path=root_path)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

        print('')
        print('Testset: %d' % len(test_set))

    return train_loader, validation_loader, test_loader

if __name__ =='__main__':

    root_path = 'Z:\data\Face\imdb_crop'
    batch_size = 20
    shuffle = True

    train, val, test = build_dataset(root_path, batch_size, shuffle)