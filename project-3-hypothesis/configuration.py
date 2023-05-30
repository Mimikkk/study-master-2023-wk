from dotdict import dotdict

config = dotdict(
  dataset=dotdict(
    name="Maysee/tiny-imagenet",
    train_size=500,
    test_size=200,
  ),
  dataloader=dotdict(
    batch_size=128,
    shuffle=True,
    pin_memory=True,
  ),
  generator=dotdict(),
  discriminator=dotdict(),
  optimizer=dotdict(betas=(0.5, 0.999), learning_rate=0.0002),
  training=dotdict(epochs=5),
  use_gpu=True,
)