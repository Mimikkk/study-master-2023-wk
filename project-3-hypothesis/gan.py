from dataclasses import dataclass

from torch.optim import Optimizer, Adam

from discriminator import Discriminator
from generator import Generator

@dataclass
class _discriminator(object):
  model: Discriminator
  optimizer: Optimizer

@dataclass
class _generator(object):
  model: Generator
  optimizer: Optimizer


@dataclass
class Gan(object):
  discriminator: _discriminator
  generator: _generator

  @classmethod
  def create(cls, discriminator: Discriminator, generator: Generator, learning_rate: float, betas: tuple[float, float]):
    return cls(
      _discriminator(
        discriminator,
        Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
      ),
      _generator(
        generator,
        Adam(generator.parameters(), lr=learning_rate, betas=betas)
      )
    )
