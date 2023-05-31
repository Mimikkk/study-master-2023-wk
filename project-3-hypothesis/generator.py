from torch import Tensor
from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh

class Generator(Module):
  def __init__(self, *, use_gpu: bool, scale):
    super(Generator, self).__init__()
    self.ngpu = use_gpu and 1 or 0

    self.model = Sequential(
      ConvTranspose2d(128, scale * 8, kernel_size=4, stride=1, padding=0, bias=False),
      BatchNorm2d(scale * 8),
      ReLU(True),
      ConvTranspose2d(scale * 8, scale * 4, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale * 4),
      ReLU(True),
      ConvTranspose2d(scale * 4, scale * 2, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale * 2),
      ReLU(True),
      ConvTranspose2d(scale * 2, scale, kernel_size=4, stride=2, padding=1, bias=False),
      BatchNorm2d(scale),
      ReLU(True),
      ConvTranspose2d(scale, 3, kernel_size=4, stride=2, padding=1, bias=False),
      Tanh()
    )

  def forward(self, input: Tensor):
    return self.model(input)
