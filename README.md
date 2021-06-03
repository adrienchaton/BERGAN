# BERGAN


trying to make techno with GANs, default is to train on 2sec audio clips at 16kHz = 1 bar 4/4 at 120BPM

generator and multi-scale discriminators derived from the MelGAN architecture

default working config = WGANGP_2scales_WN_WN_crop0.gin

without GP, it sucks, also avoid BN discriminator with GP


## TODO

try training at 32kHz

try other music genres with 4/4 musical structure

train as a VAE/GAN or WAE/GAN to avoid mode collapse of GAN and use deep feature reconstruction in discriminator activations


## AUDIO SAMPLES

examples of random linear interpolations with 20 points equally spaced in the generator latent space = 20 bars = 40 sec.

training data is between 5.000 and 20.000 examples of bars extracted from recordings from the "Raster Norton" label

models were trained for 48 hours on V100 GPU

todo


## GAN TRAINING

optimize the generator to sample realistic 1 bar audio of 2 sec. (120BPM) at SR=16kHz (extendable to 32kHz or 48kHz)

<p align="center">
  <img src="./figures/bergan_gan_train.jpg" width="750" title="GAN training">
</p>


## VAE/GAN or WAE/GAN TRAINING

to come


## GENERATION

sample series of 1 bar audio along a random linear interpolation and concatenate the generator outputs into a track at fixed BPM with progressive variation of rhythmic and acoustic content

<p align="center">
  <img src="./figures/bergan_interp.jpg" width="750" title="generator interpolation">
</p>

