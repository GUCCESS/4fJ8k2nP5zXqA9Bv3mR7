
import torch
import torch.nn as nn
import math


import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100000):

        super(PositionalEncoding, self).__init__()
        
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        
    def forward(self, x):

        x = x.permute(0, 1, 3, 4, 2)
        B, L, H, W, C = x.size()
        
        pos_encoding = self.pe[:H * W].view(H,W,C).unsqueeze(0).unsqueeze(0).to('cuda:0')

        
        pos_encoding = pos_encoding.expand(B, L, H, W, C)
        x = (x + pos_encoding).permute(0, 1, 4, 2, 3)

        return x


class LossAttentionModule(nn.Module):
    def __init__(self, config):
        super(LossAttentionModule, self).__init__()
        dim = config['dim']
        heads = config['heads']
        dim_head = config['dim_head']
        dropout = config['dropout']


        self.proj = nn.Sequential(
            nn.Linear(dim_head * heads * 4, dim, bias=False),
            nn.ReLU()
        )

        self.scale = dim_head ** -0.5

        self.to_qkv_old = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_qkv_new = nn.Linear(dim, dim_head * heads * 3, bias=False)


        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x_old, x_new):

        B, L, C, H, W = x_old.shape

        x_old = x_old.permute(0, 3, 4, 1, 2)
        x_new = x_new.permute(0, 3, 4, 1, 2)

        q_old, k_old, v_old = self.to_qkv_old(x_old).chunk(3, dim=-1)
        q_new, k_new, v_new = self.to_qkv_new(x_new).chunk(3, dim=-1)

        
        self_old_attn = torch.einsum('bhwic,bhwjc->bhwij', q_old, k_old) * self.scale
        self_new_attn = torch.einsum('bhwic,bhwjc->bhwij', q_new, k_new) * self.scale
        self_old2new_attn = torch.einsum('bhwic,bhwjc->bhwij', q_old, k_new) * self.scale
        self_new2old_attn = torch.einsum('bhwic,bhwjc->bhwij', q_new, k_old) * self.scale

        self_old_attn = self_old_attn.softmax(dim=-1)
        self_new_attn = self_new_attn.softmax(dim=-1)
        self_old2new_attn = self_old2new_attn.softmax(dim=-1)
        self_new2old_attn = self_new2old_attn.softmax(dim=-1)

        self_old_attn = torch.einsum('bhwij,bhwjc->bhwic', self_old_attn, v_old)
        self_new_attn = torch.einsum('bhwij,bhwjc->bhwic', self_new_attn, v_new)
        self_old2new_attn = torch.einsum('bhwij,bhwjc->bhwic', self_old2new_attn, v_new)
        self_new2old_attn = torch.einsum('bhwij,bhwjc->bhwic', self_new2old_attn, v_old)

        attn_output = torch.cat((self_old_attn, self_new_attn, self_old2new_attn, self_new2old_attn), dim=-1)

        attn_output = attn_output.permute(0, 3, 1, 2, 4)
        attn_output = self.proj(attn_output)

        attn_output = self.to_out(attn_output) + x_new

        return attn_output



class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        B, C, L, H, W = x.size()

        proj_query = self.query_conv(x).permute(0, 2, 1, 3, 4).reshape(B * L, -1,  H * W)

        proj_key = self.key_conv(x).permute(0, 2, 1, 3, 4).reshape(B * L, -1,  H * W).permute(0, 2, 1)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)
        proj_value = self.value_conv(x).permute(0, 2, 1, 3, 4).reshape(B * L, -1,  H * W).permute(0, 2, 1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, L, H, W, C)
        x = self.gamma * out + x.permute(0, 2, 3, 4, 1)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1))
        self.key_conv = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1))
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, prev_features):
        x = x.permute(0, 4, 1, 2, 3)
        prev_features = prev_features.permute(0, 4, 1, 2, 3)
        B, C, L, H, W = x.size()

        proj_query = self.query_conv(x).permute(0, 2, 1, 3, 4).reshape(B * L, -1, H * W)

        proj_key = self.key_conv(prev_features).permute(0, 2, 1, 4, 3).reshape(B * L, -1, H * W).permute(0, 2, 1)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)
        proj_value = self.value_conv(prev_features).permute(0, 1, 2, 3, 4).reshape(B * L, -1, H * W).permute(0, 2, 1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, L, H, W, C)
        x = self.gamma * out + x.permute(0, 2, 3, 4, 1)
        return x


class LossRecovery(nn.Module):
    
    def __init__(self, channel, config):
        super().__init__()
        self.last_time = None
        
        self.config = config
        
        self.spatial_attn = SpatialAttention(channel)
        
        self.temporal_attn = TemporalAttention(channel)
        
        self.alpha = config['temporal_attn']['alpha']
        
        
    def forward(self, x, mask, spatial_correction_matrix):


        meta_pri = x[..., -3:]
        x = x[..., :-3]
        if self.last_time == None:
            self.last_time = x.detach()
        

        x = self.spatial_attn(x)
        x = self.temporal_attn(x, self.last_time)
        
        self.last_time = self.alpha * x.detach() + (1 - self.alpha) * self.last_time

        return x

    def update_last_feature(self, new_other_feature):
        
        self.last_time = self.alpha * new_other_feature.detach() + (1 - self.alpha) * self.last_time
        
        

        
            
        
    