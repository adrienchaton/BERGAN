#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:20:19 2021

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
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from __nn_utils import Identity,Discriminator_WN,Discriminator_BN,Discriminator_LN,Discriminator_SN,init_weights,Generator,enable_disc_disable_gen,enable_gen_disable_disc,apply_zero_grad


## ADDED UTILS FOR THE AEs+GANs TRAINING


###############################################################################
## ENCODER (based on the discriminator architecture)

# MMD regularizer https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py
# https://github.com/1Konny/WAE-pytorch/blob/master/ops.py

def mmd(z_tilde, z, z_var):
    r"""Calculate maximum mean discrepancy described in the WAE paper.
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

    return out


def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


@gin.configurable
class MultiScaleEncoder(nn.Module):
    def __init__(self,encoder_norm="BN",encoder_scales=3,z_dim=128,n_hiddens=2,regularizer="none",target_sr=16000,target_dur=2.):
        super(MultiScaleEncoder, self).__init__()
        
        print("\nbuilding encoder with config = ",encoder_norm,encoder_scales,z_dim,n_hiddens,regularizer)
        
        self.regularizer = regularizer # either "none", "vae" or "wae"
        self.z_dim = z_dim
        
        if encoder_norm=="WN":
            self.encoders = nn.ModuleList(
                [Discriminator_WN() for _ in range(encoder_scales)]
            )
        if encoder_norm=="BN":
            self.encoders = nn.ModuleList(
                [Discriminator_BN() for _ in range(encoder_scales)]
            )
        if encoder_norm=="SN":
            self.encoders = nn.ModuleList(
                [Discriminator_SN() for _ in range(encoder_scales)]
            )
        
        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, encoder_scales)]
        )
        
        if encoder_norm=="LN":
            encoders = []
            with torch.no_grad():
                dummy_x = torch.rand((2,1,int(target_sr*target_dur)))
                for pool in self.pooling:
                    dummy_x = pool(dummy_x)
                    encoders.append(Discriminator_LN(dummy_x.shape[2]))
            self.encoders = nn.ModuleList(encoders)
        
        self.apply(init_weights)
        
        # MLP to map to latents
        dummy_latents = self.forward_conv(torch.zeros(1,1,int(target_sr*target_dur)))
        self.flatten_dim = dummy_latents.shape[1]
        print("multiscale flatten output of size",self.flatten_dim)
        dense_net = []
        for i in range(n_hiddens):
            if encoder_norm=="WN":
                dense_net.append(nn.utils.weight_norm(nn.Linear(self.flatten_dim,self.flatten_dim)))
            if encoder_norm=="BN":
                dense_net.append(nn.Linear(self.flatten_dim,self.flatten_dim))
                dense_net.append(nn.BatchNorm1d(self.flatten_dim))
            if encoder_norm=="SN":
                dense_net.append(nn.utils.spectral_norm(nn.Linear(self.flatten_dim,self.flatten_dim)))
            if encoder_norm=="LN":
                dense_net.append(nn.Linear(self.flatten_dim,self.flatten_dim))
                dense_net.append(nn.LayerNorm(self.flatten_dim))
            dense_net.append(nn.LeakyReLU(0.2))
        self.dense_net = nn.Sequential(*dense_net)
        
        self.mu_z = nn.Linear(self.flatten_dim,self.z_dim)
        if self.regularizer=="vae":
            self.logvar_z = nn.Linear(self.flatten_dim,self.z_dim) # TODO: could clip the log variance e.g. nn.Hardtanh(min_val=-10,max_val=2)
            
    def remove_weight_norm(self):
        # TODO: check if that works in the modulelist of the multi-scale architecture
        for idx, layer in enumerate(self.encoders):
            try:
                nn.utils.remove_weight_norm(layer)
            except:
                pass
            try:
                layer.remove_weight_norm()
            except:
                pass
        for idx, layer in enumerate(self.dense_net):
            try:
                nn.utils.remove_weight_norm(layer)
            except:
                pass
            try:
                layer.remove_weight_norm()
            except:
                pass

    def forward_conv(self, x):
        hiddens = list()

        for pool, disc in zip(self.pooling, self.encoders):
            x = pool(x)
            hiddens.append(disc(x)[1].squeeze(1))
        
        hiddens = torch.cat(hiddens,1)

        return hiddens
    
    def forward(self,x):
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        
        hiddens = self.dense_net(self.forward_conv(x))
        latents = dict()
        
        if self.regularizer=="none" or self.regularizer=="wae":
            z = self.mu_z(hiddens)
            latents["z"] = z
            if self.regularizer=="wae":
                reg = self.get_mmd(z)
                latents["reg"] = reg
        
        if self.regularizer=="vae":
            mu_z = self.mu_z(hiddens)
            logvar_z = self.logvar_z(hiddens)
            std = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std)
            z = eps * std + mu_z
            reg = self.get_kld(mu_z,logvar_z)
            if self.training:
                latents["z"] = z
            else:
                latents["z"] = mu_z # no sampling
            latents["mu_z"] = mu_z
            latents["logvar_z"] = logvar_z
            latents["reg"] = reg
        
        return latents
    
    def get_kld(self,mu_z,logvar_z):
        batch_size = mu_z.shape[0]
        mu_z = mu_z.view(batch_size,-1)
        logvar_z = logvar_z.view(batch_size,-1)
        mu_prior, logvar_prior = (torch.zeros_like(mu_z, device=mu_z.device), torch.zeros_like(logvar_z, device=logvar_z.device))
        # kld_loss = -0.5 * torch.sum(1 + logvar_z.view(batch_size,-1) - mu_z.view(batch_size,-1) ** 2 - logvar_z.exp().view(batch_size,-1), dim = 1)
        kld_loss = 0.5 * torch.sum(logvar_prior - logvar_z + torch.exp(logvar_z-logvar_prior) + torch.pow(mu_z-mu_prior,2)/torch.exp(logvar_prior) - 1, dim = 1)
        kld_loss = torch.mean(kld_loss, dim = 0)
        return kld_loss
    
    def get_mmd(self,z):
        batch_size = z.shape[0]
        z = z.view(batch_size,-1)
        z_prior = torch.randn_like(z)
        mmd_loss = mmd(z, z_prior, 1.)
        return mmd_loss


@gin.configurable
class AE(nn.Module):
    def __init__(self,encoder_norm="BN",encoder_scales=3,z_dim=220,e_n_h=2,regularizer="none",target_sr=16000,target_dur=2.,d_dense_chan_out=16,d_strides=[4,4,4,4,4],
                 d_hiddens=1024,d_n_h=1,decoder_norm="BN",d_chan_in=512,crop_steps=0,dp=0.):
        super(AE, self).__init__()
        
        self.encoder = MultiScaleEncoder(encoder_norm=encoder_norm,encoder_scales=encoder_scales,z_dim=z_dim,n_hiddens=e_n_h,
                                         regularizer=regularizer,target_sr=target_sr,target_dur=target_dur)
        
        self.decoder = Generator(z_prior="normal",dense_chan_out=d_dense_chan_out,target_sr=target_sr,z_dim=z_dim,target_dur=target_dur,strides=d_strides,
                 dense_hidden=d_hiddens,n_dense=d_n_h,dense_norm=decoder_norm,conv_norm=decoder_norm,chan_in=d_chan_in,crop_steps=crop_steps,dp=dp)
    
    def encode(self,x):
        latents = self.encoder.forward(x)
        return latents
    
    def decode(self,z):
        x = self.decoder.forward(z.shape[0], z=z)
        return x
    
    def forward(self,x):
        latents = self.encode(x)
        x_hat = self.decode(latents["z"])
        return x_hat,latents



###############################################################################
## optimization and losses

@gin.configurable
class stft_dist(nn.Module):
    def __init__(self, total_weight=0.1, log_weight=1., scales = [4096, 2048, 1024, 512, 256, 128], overlap = 0.75):
        super(stft_dist, self).__init__()
        self.total_weight = total_weight
        self.log_weight = log_weight
        self.scales = scales
        self.overlap = overlap
        print("stft reconstruction loss with parameters")
        print("total_weight,log_weight,scales,overlap",total_weight,log_weight,scales,overlap)
    
    def multiscale_fft(self,signal, scales, overlap):
        stfts = []
        for s in scales:
            S = torch.stft(
                signal,
                s,
                int(s * (1 - overlap)),
                s,
                torch.hann_window(s).to(signal),
                True,
                normalized=True,
                return_complex=True,
            ).abs()
            stfts.append(S)
        return stfts
    
    def safe_log(self,x, eps=1e-7):
        return torch.log(x + eps)
    
    def forward(self,x_inp,x_tar):
        ori_stft = self.multiscale_fft(x_tar,self.scales,self.overlap)
        rec_stft = self.multiscale_fft(x_inp,self.scales,self.overlap)
        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            if self.log_weight>0:
                log_loss = (self.safe_log(s_x) - self.safe_log(s_y)).abs().mean()
                loss = loss + lin_loss + log_loss*self.log_weight
            else:
                loss = loss + lin_loss
        return loss*self.total_weight


def __ae_loss(ae,discriminator,real_audio,adv_weight=0.1,feat_match=10.,rec_loss=None,reg_weight=0.):
    loss_g_tot = 0.0
    # LSGAN loss (generator) and deep feature reconstruction
    fake_audio,latents = ae.forward(real_audio)
    disc_fake = discriminator(fake_audio)
    disc_real = discriminator(real_audio)
    loss_g = 0.0
    for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
        loss_g = loss_g+torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
        for feat_f, feat_r in zip(feats_fake, feats_real):
            loss_g = loss_g+feat_match * torch.mean(torch.abs(feat_f - feat_r))
    loss_g_tot = loss_g_tot+loss_g*adv_weight # TODO: make adv_weight and feat_match configurable in the gin config.
    # optional spectrogram reconstruction loss
    if rec_loss is not None:
        loss_rec = rec_loss.forward(fake_audio,real_audio.squeeze(1))
        loss_g_tot = loss_g_tot+loss_rec
    else:
        loss_rec = 0.0
    # optional regularizer 
    if reg_weight>0:
        loss_reg = latents["reg"]*reg_weight
        loss_g_tot = loss_g_tot+loss_reg
    else:
        loss_reg = 0.0
    return loss_g_tot,loss_g,loss_rec,loss_reg


def __d_loss(discriminator,fake_audio,real_audio):
    # only LSGAN loss (discriminator)
    disc_fake = discriminator(fake_audio)
    disc_real = discriminator(real_audio)
    loss_d = 0.0
    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
        loss_d = loss_d+torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
        loss_d = loss_d+torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
    return loss_d




###############################################################################
## utils stuffs

def gradient_check(ae,discriminator,optim_ae,optim_d,train_dataloader,feat_match,rec_loss,target_reg_weight):
    
    for _,mb in enumerate(train_dataloader):
        break
    
    ae.train()
    discriminator.train()
    
    print('\n\ndiscriminator gradient check')
    enable_disc_disable_gen(ae,discriminator)
    real_audio = mb[0].unsqueeze(1).to(ae.decoder.device)
    fake_audio,_ = ae.forward(real_audio)
    fake_audio = fake_audio.detach()
    apply_zero_grad(ae,optim_ae,discriminator,optim_d)
    loss_d = __d_loss(discriminator,fake_audio,real_audio)
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
    
    print('\n\nae (total) gradient check')
    enable_gen_disable_disc(ae,discriminator)
    apply_zero_grad(ae,optim_ae,discriminator,optim_d)
    loss_g_tot,loss_g,loss_rec,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=feat_match,rec_loss=rec_loss,reg_weight=target_reg_weight)
    loss_g_tot.backward()
    tot_grad = 0
    named_p = ae.named_parameters()
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
    
    print('\n\nae (adversarial) gradient check')
    apply_zero_grad(ae,optim_ae,discriminator,optim_d)
    loss_g_tot,loss_g,loss_rec,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=feat_match,rec_loss=rec_loss,reg_weight=target_reg_weight)
    loss_g.backward()
    tot_grad = 0
    named_p = ae.named_parameters()
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
    
    if rec_loss is not None:
        print('\n\nstft loss gradient check')
        apply_zero_grad(ae,optim_ae,discriminator,optim_d)
        loss_g_tot,loss_g,loss_rec,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=feat_match,rec_loss=rec_loss,reg_weight=target_reg_weight)
        loss_rec.backward()
        tot_grad = 0
        named_p = ae.named_parameters()
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
    
    if target_reg_weight>0:
        print('\n\nregularization loss gradient check')
        apply_zero_grad(ae,optim_ae,discriminator,optim_d)
        loss_g_tot,loss_g,loss_rec,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=feat_match,rec_loss=rec_loss,reg_weight=target_reg_weight)
        loss_reg.backward()
        tot_grad = 0
        named_p = ae.encoder.named_parameters()
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
        
    
def export_samples(ae,real_audio,output_dir,prefix):
    bs = real_audio.shape[0]
    
    with torch.no_grad():
        fake_audio,_ = ae.forward(real_audio)
    for i in range(bs):
        output = np.concatenate([real_audio[i,0,:].cpu().numpy(),np.zeros(8000),fake_audio[i,:].cpu().numpy()])
        sf.write(output_dir+prefix+"_reconstruction_"+str(i)+".wav",output,ae.decoder.target_sr)
    
    with torch.no_grad():
        z = torch.randn((bs,ae.decoder.z_dim)).to(ae.decoder.device)
        fake_audio = ae.decode(z).cpu().numpy()
    for i in range(bs):
        sf.write(output_dir+prefix+"_randomsample_"+str(i)+".wav",fake_audio[i,:],ae.decoder.target_sr)


def export_interp(ae,n_export,n_steps,output_dir,prefix,verbose=False):
    steps = np.linspace(0.,1.,num=n_steps,endpoint=True)
    for i in range(n_export):
        with torch.no_grad():
            if verbose is True:
                print("synthesis of interp #",i)
            z = torch.randn((2,ae.decoder.z_dim)).to(ae.decoder.device)
            z_interp = []
            for s in steps:
               z_interp.append(z[0,:]*float(s)+z[1,:]*(1.-float(s)))
            z_interp = torch.stack(z_interp)
            fake_audio = ae.decode(z_interp).cpu().numpy()
        fake_audio = np.reshape(fake_audio,(-1))
        sf.write(output_dir+prefix+"_interp_"+str(i)+".wav",fake_audio,ae.decoder.target_sr)


def latent_visualiser(ae,train_dataloader,test_dataloader,save_path="./"):
    
    train_z = []
    test_z = []
    with torch.no_grad():
        for _,mb in enumerate(train_dataloader):
            train_z.append(ae.encode(mb[0].unsqueeze(1).to(ae.decoder.device))["z"].cpu().numpy())
        for _,mb in enumerate(test_dataloader):
            test_z.append(ae.encode(mb[0].unsqueeze(1).to(ae.decoder.device))["z"].cpu().numpy())
    train_z = np.concatenate(train_z,0)
    test_z = np.concatenate(test_z,0)
    
    z_ranges = np.stack([np.min(train_z,0),np.max(train_z,0)],1)
    np.save(os.path.join(save_path, "z_ranges.npy"),z_ranges)
    
    pca = PCA(n_components=2)
    pca.fit(train_z)
    train_z = pca.transform(train_z)
    test_z = pca.transform(test_z)
    
    plt.figure()
    plt.suptitle(
        "2D scatter of the training set (blue) and test set (red)")
    plt.scatter(train_z[:, 0], train_z[:, 1], c="b")
    plt.scatter(test_z[:, 0], test_z[:, 1], c="r")
    plt.savefig(os.path.join(save_path, "2D_latent_space.pdf"))








