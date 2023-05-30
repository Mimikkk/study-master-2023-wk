from torch import Tensor
from torch.nn import Sequential, Conv2d, LeakyReLU, BatchNorm2d, Sigmoid, Module, Flatten

class Discriminator(Module):
  def __init__(self, *, use_gpu: bool):
    super(Discriminator, self).__init__()
    self.ngpu = use_gpu and 1 or 0

    self.main = Sequential(
    Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    BatchNorm2d(64),
    LeakyReLU(0.2, inplace=True),
    Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    BatchNorm2d(128),
    LeakyReLU(0.2, inplace=True),
    Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    BatchNorm2d(256),
    LeakyReLU(0.2, inplace=True),
    Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    BatchNorm2d(512),
    LeakyReLU(0.2, inplace=True),
    Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    Flatten(),
    Sigmoid()
    )

  def forward(self, input: Tensor):
    return self.main(input)
