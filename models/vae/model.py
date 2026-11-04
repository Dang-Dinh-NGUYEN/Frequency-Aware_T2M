import ptwt
import pywt
import torch
import torch.nn as nn

from models.skeleton.linear import MultiLinear
from models.vae.encdec import *

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        
        self.opt = opt

        # motion encoder and decoder
        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)

        # skeleto-temporal convolutional encoder and decoder
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        # frequency enhancement module
        if getattr(opt, 'freq_enhance', False) == "dct":
            self.freq_enhanced_enc = DCTEnhancedModule(opt)
        
        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, 7)
    
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.motion_enc(x) # [B, T, 21, 32] or [B, T, 22, 32] 
        x = self.conv_enc(x) # [B, T//4, 7, 32]

        if hasattr(self, 'freq_enhanced_enc') and self.freq_enhanced_enc is not None:
            x = self.freq_enhanced_enc(x)

        # latent space
        x = self.dist(x) # [B, T//4, 7, 64]

        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)
        
        return z, {"loss_kl": loss_kl}
    
    def decode(self, z):
        x = self.conv_dec(z)
        x = self.motion_dec(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: [B, T, D]
        z: [B, T, J_out, D]
        out: [B, T, D]
        """
        # encode
        x = x.detach().float()
        z, loss_dict = self.encode(x)
        out = self.decode(z)

        return out, loss_dict