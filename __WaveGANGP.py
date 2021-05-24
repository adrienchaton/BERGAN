#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:48:32 2021

@author: adrienbitton
"""



import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import grad, Variable
import gin



## codes based on https://github.com/mostafaelaraby/wavegan-pytorch

## clip length can be [16384, 32768, 65536]
## model_size can be 32 or 64



###############################################################################
## GENERATOR LAYERS

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.bias.data.fill_(0)


class Transpose1dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        upsample=None,
        output_padding=1,
        use_batch_norm=False,
    ):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        batch_norm = nn.BatchNorm1d(out_channels)
        if self.upsample:
            operation_list = [reflection_pad, conv1d]
        else:
            operation_list = [Conv1dTrans]

        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.transpose_ops(x)


class Conv1D(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        alpha=0.2,
        shift_factor=2,
        stride=4,
        padding=11,
        use_batch_norm=False,
        drop_prob=0,
    ):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        )
        self.phase_shuffle = PhaseShuffle(shift_factor)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_phase_shuffle = shift_factor == 0
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_phase_shuffle:
            x = self.phase_shuffle(x)
        if self.use_drop:
            x = self.dropout(x)
        return x


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = (
            torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1)
            - self.shift_factor
        )
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


@gin.configurable
class WaveGANGenerator(nn.Module):
    def __init__(
        self,
        model_size=64,
        ngpus=1,
        num_channels=1,
        verbose=False,
        upsample=True,
        slice_len=32768,
        use_batch_norm=False,
        noise_latent_dim=100,
        target_sr=16000,
        target_dur=2.,
        z_prior="normal",
        ):
        super(WaveGANGenerator, self).__init__()
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances
        
        
        print("\nbuilding generator with config = ",model_size,ngpus,num_channels,verbose,upsample,
              slice_len,use_batch_norm,noise_latent_dim,target_sr,target_dur,z_prior)
        
        
        self.slice_len = slice_len
        self.target_sr = target_sr
        self.target_dur = target_dur
        self.target_len = int(target_sr*target_dur)
        self.z_prior = z_prior
        self.z_dim = noise_latent_dim

        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        latent_dim = noise_latent_dim
        self.verbose = verbose
        self.use_batch_norm = use_batch_norm

        self.dim_mul = 16 if slice_len == 16384 else 32

        self.fc1 = nn.Linear(latent_dim, 4 * 4 * model_size * self.dim_mul)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features=model_size * self.dim_mul)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4

        deconv_layers = [
            Transpose1dLayer(
                self.dim_mul * model_size,
                (self.dim_mul * model_size) // 2,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 2,
                (self.dim_mul * model_size) // 4,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 4,
                (self.dim_mul * model_size) // 8,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 8,
                (self.dim_mul * model_size) // 16,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
        ]

        if slice_len == 16384:
            deconv_layers.append(
                Transpose1dLayer(
                    (self.dim_mul * model_size) // 16,
                    num_channels,
                    25,
                    stride,
                    upsample=upsample,
                )
            )
        elif slice_len == 32768:
            deconv_layers += [
                Transpose1dLayer(
                    (self.dim_mul * model_size) // 16,
                    model_size,
                    25,
                    stride,
                    upsample=upsample,
                    use_batch_norm=use_batch_norm,
                ),
                Transpose1dLayer(model_size, num_channels, 25, 2, upsample=upsample),
            ]
        elif slice_len == 65536:
            deconv_layers += [
                Transpose1dLayer(
                    (self.dim_mul * model_size) // 16,
                    model_size,
                    25,
                    stride,
                    upsample=upsample,
                    use_batch_norm=use_batch_norm,
                ),
                Transpose1dLayer(
                    model_size, num_channels, 25, stride, upsample=upsample
                ),
            ]
        else:
            raise ValueError("slice_len {} value is not supported".format(slice_len))

        self.deconv_list = nn.ModuleList(deconv_layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, bs, z=None):
        if z is None:
            if self.z_prior=="normal":
                z = torch.randn((bs,self.z_dim)).to(self.device)
            if self.z_prior=="uniform":
                z = torch.rand((bs,self.z_dim)).to(self.device)
                z = (z*2.)-1.
        
        x = self.fc1(z).view(-1, self.dim_mul * self.model_size, 16)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        for deconv in self.deconv_list[:-1]:
            x = F.relu(deconv(x))
            if self.verbose:
                print(x.shape)
        output = torch.tanh(self.deconv_list[-1](x))
        
        if self.target_len<self.slice_len:
            output = output[:,:,:self.target_len]
            assert output.shape[2]==self.target_len
        else:
            assert output.shape[2]==self.slice_len
        
        if self.num_channels==1:
            output = output.squeeze(1)
        
        return output



###############################################################################
## DISCRIMINATOR LAYERS

@gin.configurable
class WaveGANDiscriminator(nn.Module):
    def __init__(
        self,
        model_size=64,
        ngpus=1,
        num_channels=1,
        shift_factor=2,
        alpha=0.2,
        verbose=False,
        slice_len=16384,
        use_batch_norm=False,
        ):
        super(WaveGANDiscriminator, self).__init__()
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances
        
        
        print("\nbuilding discriminator with config = ",model_size,ngpus,num_channels,shift_factor,
              alpha,verbose,slice_len,use_batch_norm)

              
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.use_batch_norm = use_batch_norm
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
            Conv1D(
                num_channels,
                model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                model_size,
                2 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                2 * model_size,
                4 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                4 * model_size,
                8 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                8 * model_size,
                16 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=0 if slice_len == 16384 else shift_factor,
            ),
        ]
        self.fc_input_size = 256 * model_size
        if slice_len == 32768:
            conv_layers.append(
                Conv1D(
                    16 * model_size,
                    32 * model_size,
                    25,
                    stride=2,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=0,
                )
            )
            self.fc_input_size = 480 * model_size
        elif slice_len == 65536:
            conv_layers.append(
                Conv1D(
                    16 * model_size,
                    32 * model_size,
                    25,
                    stride=4,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=0,
                )
            )
            self.fc_input_size = 512 * model_size

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)
        x = x.view(-1, self.fc_input_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)



###############################################################################
## optim and losses

def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def apply_zero_grad(generator,optimizer_g,discriminator,optimizer_d):
    generator.zero_grad()
    optimizer_g.zero_grad()

    discriminator.zero_grad()
    optimizer_d.zero_grad()


def enable_disc_disable_gen(generator,discriminator):
    gradients_status(discriminator, True)
    gradients_status(generator, False)


def enable_gen_disable_disc(generator,discriminator):
    gradients_status(discriminator, False)
    gradients_status(generator, True)


def disable_all(generator,discriminator):
    gradients_status(discriminator, False)
    gradients_status(generator, False)


def wavegan_d_loss(discriminator, gp_weight, generated, real, disable_GP=False):
    batch_size = real.shape[0]
    
    disc_out_gen = discriminator(generated)
    disc_out_real = discriminator(real)
    
    if disable_GP is False:
        alpha = torch.FloatTensor(batch_size, 1, 1).uniform_(0, 1).to(discriminator.device)
        alpha = alpha.expand(batch_size, real.size(1), real.size(2))
    
        interpolated = (1 - alpha) * real.data + (alpha) * generated.data[:batch_size]
        interpolated = Variable(interpolated, requires_grad=True)
    
        # calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)
        grad_inputs = interpolated
        ones = torch.ones(prob_interpolated.size()).to(discriminator.device)
        gradients = grad(
            outputs=prob_interpolated,
            inputs=grad_inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # calculate gradient penalty
        grad_penalty = (
            gp_weight
            * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        )
        assert not (torch.isnan(grad_penalty))
    else:
        grad_penalty = 0.0
    
    assert not (torch.isnan(disc_out_gen.mean()))
    assert not (torch.isnan(disc_out_real.mean()))
    cost_wd = disc_out_gen.mean() - disc_out_real.mean()
    return cost_wd, grad_penalty


def wavegan_g_loss(generator,discriminator,bs):
    generated = generator(bs).unsqueeze(1)
    discriminator_output_fake = discriminator(generated)
    loss_g = -discriminator_output_fake.mean()
    return loss_g


def gradient_check(generator,discriminator,optim_g,optim_d,gan_model,gp_weight,train_dataloader,bs):
    
    for _,mb in enumerate(train_dataloader):
        break
    
    generator.train()
    discriminator.train()
    
    print('\n\ngenerator gradient check')
    apply_zero_grad(generator,optim_g,discriminator,optim_d)
    enable_gen_disable_disc(generator,discriminator)
    loss_g = wavegan_g_loss(generator,discriminator,bs)
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
    optim_g.zero_grad()
    
    print('\n\ndiscriminator gradient check')
    enable_disc_disable_gen(generator,discriminator)
    real_audio = mb[0].unsqueeze(1).to(discriminator.device)
    fake_audio = generator(bs).unsqueeze(1)
    apply_zero_grad(generator,optim_g,discriminator,optim_d)
    loss_d,_ = wavegan_d_loss(discriminator, gp_weight, fake_audio, real_audio, disable_GP=False)
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
    optim_d.zero_grad()
    
    print('\n\ndiscriminator GP gradient check')
    enable_disc_disable_gen(generator,discriminator)
    real_audio = mb[0].unsqueeze(1).to(discriminator.device)
    fake_audio = generator(bs).unsqueeze(1)
    apply_zero_grad(generator,optim_g,discriminator,optim_d)
    _,loss_gp = wavegan_d_loss(discriminator, gp_weight, fake_audio, real_audio, disable_GP=False)
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
    optim_d.zero_grad()





"""
## forward in discriminator is compatible with cropping generator output to 32000
slice_len = 32768
target_len = 32000
bs = 3


generator = WaveGANGenerator(
        model_size=64,
        ngpus=1,
        num_channels=1,
        verbose=False,
        upsample=True,
        slice_len=slice_len,
        use_batch_norm=False,
        noise_latent_dim = 100,
        target_len=target_len,
        z_prior="normal",
    )
generator.device = "cpu"


discriminator = WaveGANDiscriminator(model_size=64,
        ngpus=1,
        num_channels=1,
        shift_factor=2,
        alpha=0.2,
        verbose=False,
        slice_len=slice_len,
        use_batch_norm=False,
    )
discriminator.device = "cpu"


print("\ngenerator forward")
x = generator(bs)
print("output shape",x.shape)

print("\ndiscriminator forward")
y = discriminator(x.unsqueeze(1))
print("output shape",y.shape)

"""







