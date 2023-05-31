from dataclasses import dataclass, field
from typing import ClassVar

from PIL.Image import Image
from datasets import load_dataset
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class DsDataset(TorchDataset):
  def __init__(self, data, targets, transform=None):
    self.data = data
    self.targets = LongTensor(targets)
    self.transform = transform

  def __getitem__(self, index):
    x = self.data[index]
    y = self.targets[index]

    if self.transform:
      x = self.transform(x)

    return x, y

  def __len__(self):
    return len(self.data)

@dataclass
class Dataset(object):
  train: list[tuple[Image, int]] = field(default_factory=list)
  test: list[tuple[Image, int]] = field(default_factory=list)
  _transform: ClassVar[Compose] = Compose([
    Resize(64),
    CenterCrop(64),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  @classmethod
  def load(cls, *, name: str, train_size: int = 1, test_size: int = 1):
    train, test = load_dataset(
      name,
      split=[f'train[:{train_size}]', f'valid[:{test_size}]']
    )

    train = [(row['image'], row['label']) for row in train]
    test = [(row['image'], row['label']) for row in test]

    return cls(train, test)

  def dataloader(self, *, type: str, batch_size: int, shuffle: bool, pin_memory: bool):
    X, y = zip(*self.__dict__[type])

    return DataLoader(
      DsDataset(X, y, transform=self._transform),
      batch_size=batch_size,
      shuffle=shuffle,
      pin_memory=pin_memory
    )
