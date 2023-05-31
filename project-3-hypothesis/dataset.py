from dataclasses import dataclass, field
from typing import ClassVar

from PIL.Image import Image
import PIL.Image as ImageMethods
from datasets import load_dataset
import numpy as np
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from numpy import min, max, log, abs, stack, uint8 as u8, asarray
from numpy.fft import fft2, fftshift, ifft, ifftshift


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

  def dataloader(self, *, domain: str, type: str, batch_size: int, shuffle: bool, pin_memory: bool):
    if domain == 'time':
      return self._time_domain_dataloader(
        type=type,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory
      )
    if domain == 'frequency':
      return self._frequency_domain_dataloader(
        type=type,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory
      )

  def _time_domain_dataloader(self, *, type: str, batch_size: int, shuffle: bool, pin_memory: bool):
    X, y = zip(*self.__dict__[type])

    return DataLoader(
      DsDataset(X, y, transform=self._transform),
      batch_size=batch_size,
      shuffle=shuffle,
      pin_memory=pin_memory
    )

  def _frequency_domain_dataloader(self, *, type: str, batch_size: int, shuffle: bool, pin_memory: bool):
    X, y = zip(*self.__dict__[type])

    images = []
    for image in X:
      image = np.asarray(image)

      red_fft = fft2(image[:, :, 0])
      green_fft = fft2(image[:, :, 1])
      blue_fft = fft2(image[:, :, 2])

      red_shift = fftshift(red_fft)
      green_shift = fftshift(green_fft)
      blue_shift = fftshift(blue_fft)

      red_magnitude = log(abs(red_shift) + 1)
      green_magnitude = log(abs(green_shift) + 1)
      blue_magnitude = log(abs(blue_shift) + 1)
      red_magnitude = (
          (red_magnitude - min(red_magnitude)) / (max(red_magnitude) - min(red_magnitude))
      )
      green_magnitude = (
          (green_magnitude - min(green_magnitude)) / (max(green_magnitude) - min(green_magnitude))
      )
      blue_magnitude = (
          (blue_magnitude - min(blue_magnitude)) / (max(blue_magnitude) - min(blue_magnitude))
      )

      images.append(
        ImageMethods.fromarray(
          (stack([red_magnitude, green_magnitude, blue_magnitude], axis=2) * 255).astype(u8),
          mode='RGB'
        )
      )

    return DataLoader(
      DsDataset(images, y, transform=self._transform),
      batch_size=batch_size,
      shuffle=shuffle,
      pin_memory=pin_memory
    )
