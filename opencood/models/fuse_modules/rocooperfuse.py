

import torch
import torch.nn as nn
from opencood.models.sub_modules.loss_rocovery import *
from opencood.models.sub_modules.base_transformer import *
from opencood.models.fuse_modules.select_attention import *
from opencood.models.fuse_modules.windowAttention import WindowAttention

class RoCooperEncoderBlock(nn.Module):
    def __init__(self, args):
        super(RoCooperEncoderBlock, self).__init__()
        self.layers = nn.ModuleList([])
        self.cross_atten_config = args['cross_windows']
        self.p_self_atten_config = args['p_self_atten_config']
        
        self.layers.append(nn.ModuleList([
            PreNorm(args['dim'], CrossAttentionWindow(self.cross_atten_config)),
            PreNorm(args['dim'], WindowAttention(dim = self.p_self_atten_config['dim'],\
                heads=self.p_self_atten_config['heads'],\
                    dim_heads = self.p_self_atten_config['dim_heads'],\
                        drop_out = self.p_self_atten_config['dropout'],\
                            window_size = self.p_self_atten_config['window_size'],\
                                relative_pos_embedding = self.p_self_atten_config['relative_pos_embedding'],\
                                    fuse_method = 'split_attn'))
        ]))
            
        
    def forward(self, x):
        
        for cross_atten , window_self_attn in self.layers:
            x = cross_atten(x) + x
            x = window_self_attn(x) + x
            
        return x


class RoCooperEncoder(nn.Module):
    
    def __init__(self, args):
        super(RoCooperEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.loss_r_config = args['loss_recovery']
        self.feed_forward_config = args['feed_forward']
        self.loss_recovery_module = LossRecovery(self.loss_r_config['dim'], self.loss_r_config)
        self.fusion_block_config = args['fusion']
        
        
        for _ in range(args['depth']):
            self.layers.append(nn.ModuleList([ RoCooperEncoderBlock(self.fusion_block_config),  PreNorm(self.fusion_block_config['dim'], FeedForward(self.fusion_block_config['dim'], self.feed_forward_config['mlp_dim'],\
                            dropout= self.feed_forward_config['dropout']))
            ]))
            
            
    def forward(self, x, mask, spatial_correction_matrix):
        

        if self.loss_r_config['enable'] == 1:
            x = self.loss_recovery_module(x, mask, spatial_correction_matrix)
        else:
            x = x[..., :-3]

        for encoder, f in self.layers:
            x = encoder(x)
            x = f(x) + x

        return x
        
               
        

class RoCooperFusion(nn.Module):
    
    def __init__(self, args):
        super(RoCooperFusion, self).__init__()
        self.encoder = RoCooperEncoder(args['encoder'])
        
    def forward(self, x, mask, spatial_correction_matrix):
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]

        return output