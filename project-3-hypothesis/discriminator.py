from torch import Tensor
from torch.nn import Sequential, Conv2d, LeakyReLU, BatchNorm2d, Sigmoid, Module, Flatten

class Discriminator(Module):
  def __init__(self, *, use_gpu: bool, scale: int):
    super(Discriminator, self).__init__()
    self.ngpu = use_gpu and 1 or 0

    self.main = Sequential(
      Conv2d(3, scale, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale),
      LeakyReLU(0.2, inplace=True),
      Conv2d(scale, scale * 2, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale * 2),
      LeakyReLU(0.2, inplace=True),
      Conv2d(scale * 2, scale * 4, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale * 4),
      LeakyReLU(0.2, inplace=True),
      Conv2d(scale * 4, scale * 8, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale * 8),
      LeakyReLU(0.2, inplace=True),
      Conv2d(scale * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
      Flatten(),
      Sigmoid()
    )

  def forward(self, input: Tensor):
    return self.main(input)
