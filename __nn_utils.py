#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:39:50 2021

@author: adrienbitton
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch import autograd
import numpy as np
import glob
import random
import os
import soundfile as sf
import librosa
import gin



## nn stuffs
## based on https://github.com/seungwonpark/melgan/blob/master/model/generator.py



###############################################################################
## GENERATOR LAYERS

def init_weights(m):
    # all layers are either conv, transposed conv or linear with leaky relu and slope "a"
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        # nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


class ResStack(nn.Module):
    def __init__(self, channel,conv_norm):
        super(ResStack, self).__init__()
        
        if conv_norm=="WN":
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad1d(3**i),
                    nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                    nn.LeakyReLU(0.2),
                    nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)))
            for i in range(3)
            ])
            self.shortcuts = nn.ModuleList([
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(3)
            ])
        elif conv_norm=="BN":
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad1d(3**i),
                    nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i),
                    nn.BatchNorm1d(channel),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(channel, channel, kernel_size=1),
                    nn.BatchNorm1d(channel))
            for i in range(3)
            ])
            self.shortcuts = nn.ModuleList([
                nn.Sequential(nn.Conv1d(channel, channel, kernel_size=1),
                          nn.BatchNorm1d(channel))
            for i in range(3)
            ])
        else:
            raise ValueError('normalization argument not valid')
    
    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x
    
    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            try:
                nn.utils.remove_weight_norm(block[2])
                nn.utils.remove_weight_norm(block[4])
                nn.utils.remove_weight_norm(shortcut)
            except:
                pass


@gin.configurable
class Generator(nn.Module):
    def __init__(self, z_prior="normal",dense_chan_out=16,target_sr=16000,z_dim=128,target_dur=2.,strides=[8,8,2,2],
                 dense_hidden=1024,n_dense=1,dense_norm="WN",conv_norm="WN",chan_in=512,crop_steps=10,dp=0.):
        super(Generator, self).__init__()
        
        print("\nbuilding generator with config = ",z_prior,dense_chan_out,target_sr,z_dim,target_dur,strides,
                 dense_hidden,n_dense,dense_norm,conv_norm,chan_in,crop_steps,dp)
        
        self.z_prior = z_prior
        self.dense_chan_out = dense_chan_out
        self.target_sr = target_sr
        self.z_dim = z_dim
        self.target_dur = target_dur
        self.strides = strides
        self.crop_steps = crop_steps
        
        self.n_upconv = len(strides)
        self.hop_length = np.prod(strides)
        self.target_len = int(self.target_sr*self.target_dur)
        self.target_frames = int(np.ceil(self.target_len/self.hop_length))
        self.uncrop_len = self.hop_length*self.target_frames
        self.dense_out = self.dense_chan_out*self.target_frames
        
        dense_net = []
        for i in range(n_dense):
            if n_dense==1:
                h_in = z_dim
                h_out = self.dense_out
            else:
                if i==0:
                    h_in = z_dim
                    h_out = dense_hidden
                elif i==n_dense-1:
                    h_in = dense_hidden
                    h_out = self.dense_out
                else:
                    h_in = dense_hidden
                    h_out = dense_hidden
            if dense_norm=="WN":
                dense_net += [nn.utils.weight_norm(nn.Linear(h_in,h_out))]
            elif dense_norm=="BN":
                dense_net += [nn.Linear(h_in,h_out),nn.BatchNorm1d(h_out)]
            elif dense_norm=="LN":
                dense_net += [nn.Linear(h_in,h_out),nn.LayerNorm(h_out)]
            elif dense_norm=="NONE":
                dense_net += [nn.Linear(h_in,h_out)]
            else:
                raise ValueError('normalization argument not valid')
            dense_net.append(nn.LeakyReLU(0.2))
            if dp>0:
                dense_net.append(nn.Dropout(p=dp))
        self.dense_net = nn.Sequential(*dense_net)
        
        print("\ndense mapping from z_dim to dense_chan_out*target_frames = dense_out",z_dim,self.dense_chan_out,self.target_frames,self.dense_out)
        # print(self.dense_net)
        
        if conv_norm=="WN":
            generator = [nn.ReflectionPad1d(3),
                         nn.utils.weight_norm(nn.Conv1d(self.dense_chan_out, chan_in, kernel_size=7, stride=1)),
                         nn.LeakyReLU(0.2)]
            for i in range(self.n_upconv):
                generator += [nn.utils.weight_norm(nn.ConvTranspose1d(chan_in//(2**i), chan_in//(2**(i+1)),
                                                        kernel_size=16, stride=self.strides[i], padding=self.strides[i]//2)),
                              ResStack(chan_in//(2**(i+1)),conv_norm),
                              nn.LeakyReLU(0.2)]
            generator += [nn.ReflectionPad1d(3),
                          nn.utils.weight_norm(nn.Conv1d(chan_in//(2**(i+1)), 1, kernel_size=7, stride=1)),
                          nn.Tanh()]
            self.generator = nn.Sequential(*generator)
            
        elif conv_norm=="BN":
            generator = [nn.ReflectionPad1d(3),
                         nn.Conv1d(self.dense_chan_out, chan_in, kernel_size=7, stride=1),
                         nn.BatchNorm1d(chan_in),
                         nn.LeakyReLU(0.2)]
            for i in range(self.n_upconv):
                generator += [nn.ConvTranspose1d(chan_in//(2**i), chan_in//(2**(i+1)),
                                                        kernel_size=16, stride=self.strides[i], padding=self.strides[i]//2),
                              nn.BatchNorm1d(chan_in//(2**(i+1))),
                              ResStack(chan_in//(2**(i+1)),conv_norm),
                              nn.LeakyReLU(0.2)]
            generator += [nn.ReflectionPad1d(3),
                          nn.Conv1d(chan_in//(2**(i+1)), 1, kernel_size=7, stride=1),
                          nn.Tanh()]
            self.generator = nn.Sequential(*generator)
            
        else:
            raise ValueError('normalization argument not valid')
        
        print("\ngenerator mapping from chan_in to chan_out",chan_in,chan_in//(2**(i+1)))
        
        print("\noutput with upsampling, uncrop_len and target_len =",self.hop_length,self.uncrop_len,self.target_len)
        # print(self.generator)
        
        self.apply(init_weights)
    
    def forward(self, bs, z=None):
        if z is None:
            if self.z_prior=="normal":
                z = torch.randn((bs,self.z_dim)).to(self.device)
            if self.z_prior=="uniform":
                z = torch.rand((bs,self.z_dim)).to(self.device)
                z = (z*2.)-1.
        h_dense = self.dense_net(z).view(-1,self.dense_chan_out,self.target_frames)
        if self.crop_steps>0:
            h_dense = torch.cat((h_dense,torch.zeros(bs,self.dense_chan_out,self.crop_steps).to(h_dense.device)),dim=2)
        audio = self.generator(h_dense).squeeze(1)
        if audio.shape[1]>self.target_len:
            audio = audio[:,:self.target_len]
        assert audio.shape[1]==self.target_len
        return audio
    
    def remove_weight_norm(self):
        for idx, layer in enumerate(self.dense_net):
            try:
                nn.utils.remove_weight_norm(layer)
            except:
                pass
            try:
                layer.remove_weight_norm()
            except:
                pass
        for idx, layer in enumerate(self.generator):
            try:
                nn.utils.remove_weight_norm(layer)
            except:
                pass
            try:
                layer.remove_weight_norm()
            except:
                pass


# should take care of removing weight norm during inference after training

# see https://github.com/seungwonpark/melgan/issues/8
# inference with zero padding at the end to avoid a click at the end



###############################################################################
## DISCRIMINATOR LAYERS

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Discriminator_WN(nn.Module):
    def __init__(self):
        super(Discriminator_WN, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


class Discriminator_BN(nn.Module):
    def __init__(self):
        super(Discriminator_BN, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.Conv1d(1, 16, kernel_size=15, stride=1),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
        ]) # the last discriminator layer is not normalized and doesnt have an activation

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


## cf. https://github.com/andreaferretti/wgan/blob/master/models.py ; critic uses LN
## LN in the convolution requires giving the channel and length dimensions
class Discriminator_LN(nn.Module):
    def __init__(self,scale_length):
        super(Discriminator_LN, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.Conv1d(1, 16, kernel_size=15, stride=1),
            ),
            nn.Sequential(
                nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            ),
            nn.Sequential(
                nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            ),
            nn.Sequential(
                nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
            ),
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
        ]) # the last discriminator layer is not normalized and doesnt have an activation
        
        norms = []
        with torch.no_grad():
            dummy_x = torch.rand((2,1,scale_length))
            for module in self.discriminator[:-1]:
                dummy_x = module(dummy_x)
                norms.append(nn.Sequential(nn.LayerNorm((dummy_x.shape[1],dummy_x.shape[2])),nn.LeakyReLU(0.2, inplace=True)))
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for i,module in enumerate(self.discriminator):
            x = module(x)
            if i<len(self.discriminator)-1:
                x = self.norms[i](x)
            features.append(x)
        return features[:-1], features[-1]


# note: there is TORCH.NN.UTILS.REMOVE_SPECTRAL_NORM but we dont need it as the discriminator is only used for training
class Discriminator_SN(nn.Module):
    def __init__(self):
        super(Discriminator_SN, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.spectral_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.spectral_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


@gin.configurable
class MultiScaleDiscriminator(nn.Module):
    def __init__(self,disc_norm="WN",disc_scales=3,target_sr=16000,target_dur=2.):
        super(MultiScaleDiscriminator, self).__init__()
        
        print("\nbuilding discriminator with config = ",disc_norm,disc_scales)
        
        if disc_norm=="WN":
            self.discriminators = nn.ModuleList(
                [Discriminator_WN() for _ in range(disc_scales)]
            )
        if disc_norm=="BN":
            self.discriminators = nn.ModuleList(
                [Discriminator_BN() for _ in range(disc_scales)]
            )
        if disc_norm=="SN":
            self.discriminators = nn.ModuleList(
                [Discriminator_SN() for _ in range(disc_scales)]
            )
        
        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, disc_scales)]
        )
        
        if disc_norm=="LN":
            discriminators = []
            with torch.no_grad():
                dummy_x = torch.rand((2,1,int(target_sr*target_dur)))
                for pool in self.pooling:
                    dummy_x = pool(dummy_x)
                    discriminators.append(Discriminator_LN(dummy_x.shape[2]))
            self.discriminators = nn.ModuleList(discriminators)
        
        self.apply(init_weights)

    def forward(self, x):
        ret = list()

        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))

        return ret # [(feat, score), (feat, score), (feat, score)]



###############################################################################
## optimization and losses

def apply_zero_grad(generator,optimizer_g,discriminator,optimizer_d):
    generator.zero_grad()
    optimizer_g.zero_grad()

    discriminator.zero_grad()
    optimizer_d.zero_grad()


def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def enable_disc_disable_gen(generator,discriminator):
    gradients_status(discriminator, True)
    gradients_status(generator, False)


def enable_gen_disable_disc(generator,discriminator):
    gradients_status(discriminator, False)
    gradients_status(generator, True)


def disable_all(generator,discriminator):
    gradients_status(discriminator, False)
    gradients_status(generator, False)


# The discriminator score output is of shape [bs, 1, t]
# convention adopted is to mean dim0 and sum dim2


# WGAN-GP: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
# https://github.com/andreaferretti/wgan
# https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
# no batch-norm and no sigmoid output in discriminator
# aim is to have gradients with norm at most 1 everywhere --> perform interpolation between the real and generated data
# discriminator returns scalar score rather than a probability


def gradient_penalty(discriminator,gp_weight,real_data,generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data).to(discriminator.device)
    interpolated = alpha * real_data.detach() + (1 - alpha) * generated_data.detach()
    interpolated = interpolated.to(discriminator.device)
    interpolated.requires_grad_(True)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)
    
    loss_gp = 0.0
    for _, prob in prob_interpolated:
        prob = prob.squeeze(1)
        
        gradients = autograd.grad(outputs=prob,
                                inputs=interpolated,
                                grad_outputs=torch.ones(prob.size()).to(discriminator.device),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    
        # flatten to easily take norm per example in batch
        # gradients = gradients.view(batch_size, -1)
        
        # gradient_norm = gradients.norm(2, dim=1).mean()
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    
        # Return gradient penalty
        loss_gp = loss_gp + gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    return loss_gp


def __g_loss(gan_model,generator,discriminator,bs):
    fake_audio = generator(bs).unsqueeze(1)
    disc_fake = discriminator(fake_audio)
    loss_g = 0.0
    for _, score_fake in disc_fake:
        if gan_model=="WGANGP":
            loss_g = loss_g-torch.mean(torch.sum(score_fake, dim=[1, 2]))
            # loss_g = loss_g-score_fake.mean()
        if gan_model=="LSGANGP":
            loss_g = loss_g+torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
    return loss_g


def __d_loss(gan_model,discriminator,gp_weight,fake_audio,real_audio,disable_GP=False):
    disc_fake = discriminator(fake_audio)
    disc_real = discriminator(real_audio)
    loss_d = 0.0
    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
        if gan_model=="WGANGP":
            loss_d = loss_d-torch.mean(torch.sum(score_real, dim=[1, 2]))
            # loss_d = loss_d-score_real.mean()
            loss_d = loss_d+torch.mean(torch.sum(score_fake, dim=[1, 2]))
            # loss_d = loss_d+score_fake.mean()
        if gan_model=="LSGANGP":
            loss_d = loss_d+torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
            loss_d = loss_d+torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
    if disable_GP is False and gp_weight is not None:
        loss_gp = gradient_penalty(discriminator,gp_weight,real_audio,fake_audio)
    else:
        # e.g. test step, cannot compute gradient and GP
        loss_gp = 0.0
    return loss_d,loss_gp


# Least-square GAN
# we dont use log-loss GAN to avoid sigmoid output and vanishing gradients because of saturation

# def lsgangp_g_loss(generator,discriminator,bs):
#     fake_audio = generator(bs).unsqueeze(1)
#     disc_fake = discriminator(fake_audio)
#     loss_g = 0.0
#     for _, score_fake in disc_fake:
#         loss_g = loss_g+torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
#     return loss_g


# def lsgangp_d_loss(discriminator,gp_weight,fake_audio,real_audio,disable_GP=False):
#     disc_fake = discriminator(fake_audio)
#     disc_real = discriminator(real_audio)
#     loss_d = 0.0
#     for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
#         loss_d = loss_d+torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
#         loss_d = loss_d+torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
#     if disable_GP is False and gp_weight is not None:
#         loss_gp = gradient_penalty(discriminator,gp_weight,real_audio,fake_audio)
#     else:
#         # e.g. test step, cannot compute gradient and GP
#         loss_gp = 0.0
#     return loss_d,loss_gp


"""
###############################################################################
## testing

bs = 2 # default is 16
lr = 0.0001
beta1 = 0.5
beta2 = 0.9
target_sr = 16000
target_dur = 2.
true_audio = torch.randn((bs,int(target_dur*target_sr)))


generator = Generator(z_prior="normal", dense_chan_out=16,target_sr=target_sr,z_dim=128,target_dur=target_dur,strides=[4,4,4,4,4],
                 dense_hidden=1024,n_dense=3,dense_norm="BN",conv_norm="BN",chan_in=512,crop_steps=10,dp=0.)
generator.device = "cpu"
discriminator = MultiScaleDiscriminator(disc_norm="LN",disc_scales=3,target_sr=target_sr,target_dur=target_dur)

generator.train()
fake_audio = generator.forward(bs) # (batchnorm only forward on bs=1 in eval mode)

disc_fake = discriminator(fake_audio.unsqueeze(1))
for _, score_fake in disc_fake:
    print(score_fake.shape)
    # torch.Size([bs, 1, 125])
    # torch.Size([bs, 1, 63])
    # torch.Size([bs, 1, 32])
"""


###############################################################################
## gradient checking

def gradient_check(generator,discriminator,optim_g,optim_d,gan_model,gp_weight,train_dataloader,bs):
    
    for _,mb in enumerate(train_dataloader):
        break
    
    generator.train()
    discriminator.train()
    
    print('\n\ngenerator gradient check')
    enable_gen_disable_disc(generator,discriminator)
    apply_zero_grad(generator,optim_g,discriminator,optim_d)
    loss_g = __g_loss(gan_model,generator,discriminator,bs)
    loss_g.backward()
    tot_grad = 0
    named_p = generator.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)
    
    print('\n\ndiscriminator gradient check')
    enable_disc_disable_gen(generator,discriminator)
    fake_audio = generator(bs).unsqueeze(1)
    fake_audio = fake_audio.detach()
    real_audio = mb[0].unsqueeze(1).to(generator.device)
    apply_zero_grad(generator,optim_g,discriminator,optim_d)
    loss_d,_ = __d_loss(gan_model,discriminator,gp_weight,fake_audio,real_audio)
    loss_d.backward()
    tot_grad = 0
    named_p = discriminator.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)
    
    if gp_weight is not None:
        print('\n\ndiscriminator GP gradient check')
        fake_audio = generator(bs).unsqueeze(1)
        fake_audio = fake_audio.detach()
        real_audio = mb[0].unsqueeze(1).to(generator.device)
        apply_zero_grad(generator,optim_g,discriminator,optim_d)
        _,loss_gp = __d_loss(gan_model,discriminator,gp_weight,fake_audio,real_audio)
        loss_gp.backward()
        tot_grad = 0
        named_p = discriminator.named_parameters()
        for name, param in named_p:
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)



###############################################################################
## utils stuffs

def load_data(wav_dir,bs,target_sr,target_dur,artists_sel):
    if len(artists_sel)==0:
        all_files = glob.glob(wav_dir+"*.wav")
        if len(all_files)==0:
            all_files = glob.glob(wav_dir+"**/*.wav")
    else:
        print("loading subsets",artists_sel)
        all_files = []
        for artist in artists_sel:
            all_files += glob.glob(wav_dir+artist+"/*.wav")
    
    random.shuffle(all_files)
    train_files = all_files[:int(len(all_files)*0.9)]
    test_files = all_files[int(len(all_files)*0.9):]
    
    train_audio = []
    for file in train_files:
        y,sr = sf.read(file)
        assert sr==target_sr
        assert len(y)==int(target_sr*target_dur)
        train_audio.append(y.astype("float32"))
    train_audio = torch.from_numpy(np.stack(train_audio)).float()
    
    test_audio = []
    for file in test_files:
        y,sr = sf.read(file)
        assert sr==target_sr
        assert len(y)==int(target_sr*target_dur)
        test_audio.append(y.astype("float32"))
    test_audio = torch.from_numpy(np.stack(test_audio)).float()
    
    train_dataset = torch.utils.data.TensorDataset(train_audio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=bs,shuffle=True,drop_last=True)
    
    test_dataset = torch.utils.data.TensorDataset(test_audio)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=bs,shuffle=True,drop_last=True)
    
    print("built dataloaders with shapes of train_audio,test_audio =",train_audio.shape,test_audio.shape)
    
    return train_dataloader,test_dataloader


def export_samples(generator,n_export,output_dir,prefix):
    with torch.no_grad():
        fake_audio = generator(n_export).cpu().numpy()
    for i in range(n_export):
        sf.write(output_dir+prefix+"_sample_"+str(i)+".wav",fake_audio[i,:],generator.target_sr)


def export_groundtruth(mb,target_sr,output_dir,prefix):
    bs = mb.shape[0]
    mb = mb.cpu().numpy()
    for i in range(bs):
        sf.write(output_dir+prefix+"_groundtruth_"+str(i)+".wav",mb[i,:],target_sr)


def print_time(s_duration):
    m,s = divmod(s_duration,60)
    h, m = divmod(m, 60)
    return h, m, s


def export_interp(generator,n_export,n_steps,output_dir,prefix,verbose=False):
    steps = np.linspace(0.,1.,num=n_steps,endpoint=True)
    for i in range(n_export):
        with torch.no_grad():
            if verbose is True:
                print("synthesis of interp #",i)
            if generator.z_prior=="normal":
                z = torch.randn((2,generator.z_dim)).to(generator.device)
            if generator.z_prior=="uniform":
                z = torch.rand((2,generator.z_dim)).to(generator.device)
                z = (z*2.)-1.
            
            z_interp = []
            for s in steps:
               z_interp.append(z[0,:]*float(s)+z[1,:]*(1.-float(s)))
            z_interp = torch.stack(z_interp)
            
            fake_audio = generator(z_interp.shape[0],z=z_interp).cpu().numpy()
        fake_audio = np.reshape(fake_audio,(-1))
        sf.write(output_dir+prefix+"_interp_"+str(i)+".wav",fake_audio,generator.target_sr)





