from torch.nn import Module
from torch.nn.init import normal_, constant_

def initialize_weights(module: Module):
  classname = module.__class__.__name__

  if classname.find('Conv') != -1:
    normal_(module.weight.data, 0.0, 0.02)
    return

  if classname.find('BatchNorm') != -1:
    normal_(module.weight.data, 1.0, 0.02)
    constant_(module.bias.data, 0)
    return
