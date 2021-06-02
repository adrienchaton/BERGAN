# BERGAN


trying to make techno with GANs, default is to train on 2sec audio clips at 16kHz = 1 bar 4/4 at 120BPM

generator and multi-scale discriminators derived from the MelGAN architecture

default working config = WGANGP_2scales_WN_WN_crop0.gin

without GP, it sucks, also avoid BN discriminator with GP

note: weirdly, discriminator's layer biases seem not to receive gradients :((

