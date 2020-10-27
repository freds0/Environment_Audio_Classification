from torchvision import datasets, transforms
from base import BaseDataLoader

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from os.path import join
from utils.audio import get_melspectrogram_db, spec_to_image
'''
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
'''
class ESC50DataLoader(Dataset):
  def __init__(self, base, filepath, in_col, out_col):
    self.df = pd.read_csv(filepath)
    self.data = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(self.df[out_col].unique())

    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category

    for ind in range(len(self.df)):
      row = self.df.iloc[ind]
      file_path = join(base,row[in_col])
      self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
      self.labels.append(self.c2i[row['category']])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]