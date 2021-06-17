#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:55:27 2021

@author: adrienbitton
"""



import torch
from __nn_utils import Generator, MultiScaleDiscriminator
from __nn_utils_ae import MultiScaleEncoder, AE, __ae_loss, __d_loss, stft_dist
import gin



gin.enter_interactive_mode()



###############################################################################
## testing the forward(s) and losses

bs = 2
target_sr = 16000
target_dur = 2.
regularizer = "vae"
if regularizer=="none":
    reg_weight = 0.
else:
    reg_weight = 1.
z_dim = 8
real_audio = torch.randn((bs,int(target_dur*target_sr)))


generator = Generator(z_prior="normal",dense_chan_out=16,target_sr=target_sr,z_dim=z_dim,target_dur=target_dur,strides=[4,4,4,4,4],
                 dense_hidden=1024,n_dense=1,dense_norm="WN",conv_norm="WN",chan_in=512,crop_steps=0,dp=0.)
generator.device = "cpu"


discriminator = MultiScaleDiscriminator(disc_norm="WN",disc_scales=3,target_sr=target_sr,target_dur=target_dur)
discriminator.device = "cpu"


encoder = MultiScaleEncoder(encoder_norm="BN",encoder_scales=3,z_dim=z_dim,n_hiddens=2,regularizer=regularizer,target_sr=target_sr,target_dur=target_dur)

ae = AE(encoder_norm="BN",encoder_scales=3,z_dim=z_dim,e_n_h=2,regularizer=regularizer,target_sr=target_sr,target_dur=target_dur,d_dense_chan_out=16,d_strides=[4,4,4,4,4],
                 d_hiddens=1024,d_n_h=1,decoder_norm="BN",d_chan_in=512,crop_steps=0,dp=0.)
ae.decoder.device = "cpu"


print("\n\ngenerator forward")
generator.train()
fake_audio = generator.forward(bs) # (batchnorm only forward on bs=1 in eval mode)


print("\n\ndiscriminator forward")
disc_fake = discriminator(fake_audio.unsqueeze(1))
for _, score_fake in disc_fake:
    print(score_fake.shape)
    # torch.Size([bs, 1, 125])
    # torch.Size([bs, 1, 63])
    # torch.Size([bs, 1, 32])


print("\n\nencoder forward")
latents = encoder(fake_audio.unsqueeze(1))
print(latents)


print("\n\nauto-encoder forward")
rec_audio,latents = ae.forward(fake_audio.unsqueeze(1))
print(latents)

rec_loss = stft_dist(total_weight=0.1, log_weight=1., scales = [4096, 2048, 1024, 512, 256, 128], overlap = 0.75)
# rec_loss = None

loss_g_tot,loss_g,loss_rec,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=10.,rec_loss=rec_loss,reg_weight=reg_weight)
print("loss_g_tot,loss_g,loss_rec,loss_reg",loss_g_tot,loss_g,loss_rec,loss_reg)

d_losses = __d_loss(discriminator,fake_audio,real_audio)






