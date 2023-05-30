from torch import Tensor
from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh

class Generator(Module):
  def __init__(self, *, use_gpu: bool):
    super(Generator, self).__init__()
    self.ngpu = use_gpu and 1 or 0

    self.model = Sequential(
      ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False),
      BatchNorm2d(512),
      ReLU(True),
      ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(256),
      ReLU(True),
      ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(128),
      ReLU(True),
      ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(64),
      ReLU(True),
      ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
      Tanh()
    )

  def forward(self, input: Tensor):
    return self.model(input)
