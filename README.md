# BERGAN

trying to make techno with GANs, intended to train on 2sec audio clips at 16kHz = 1 bar 4/4 at 120BPM

__nn_utils.py and __train.py were first done to train a MelGAN inspired model without or with gradient penalty

run with configs named "GAN" and "WGANGP"

then reused the WaveGANv2 of https://github.com/mostafaelaraby/wavegan-pytorch to compare

__WaveGANGP.py and __train_wavegan.py to run with the configs named "WAVEGAN"

so far configs that seem to give results are

WGANGP_2scales_BN_WN_crop0.gin / WGANGP_2scales_BN_WN_crop0_G1024.gin

and

WAVEGAN_default.gin / WAVEGAN_default_small.gin

without GP, it sucks, also avoid BN discriminator with GP

note: weirdly, discriminator's layer biases seem not to receive gradient :((
