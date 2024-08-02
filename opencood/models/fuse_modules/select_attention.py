import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_windows_num(window_size, H, W):

    num_windows_h = H // window_size
    num_windows_w = W // window_size
    
    return num_windows_h * num_windows_w

class CrossAttentionWindow(nn.Module):
    
    
    def __init__(self, args):
        super().__init__()
        
        dim = args['dim']
        heads = args['heads']
        dim_head = args['dim_head']
        dropout = args['dropout']  
        window_sizes = args['window_sizes']
        H = args['H'] 
        W = args['W'] 
        
        
        self.window_sizes = window_sizes
        self.attn_modules = nn.ModuleDict({
            str(window_size): CrossAttention(dim, heads[window_sizes.index(window_size)], \
                dim_head[window_sizes.index(window_size)], dropout, window_size, \
                    window_num = get_windows_num(window_size, H, W))
            for window_size in window_sizes
        })
        self.router = nn.Linear(dim, len(window_sizes))
        self.k_window_size_rate = args['k_window_size_rate']  
        self.k_window_rate = args['k_window_rate']  
        
        
        self.window_fusion_method = args['merge_method']
        self.window_fusion = SplitAttn(dim)


    def forward(self, x):
        
        
        x = x.permute(0, 1, 4, 2, 3)
        
        k_window_size = int(self.k_window_size_rate * len(self.window_sizes))  # select window_size
        
        B, L, C, H, W = x.shape  # x.shape: (B, L, C, H, W)
        
        
        
        router_compute = x[:, 0].permute(0,2,3,1).view(B, H * W, C)
        router_compute = self.router(router_compute)\
            .softmax(dim=2)\
                .view(B, len(self.window_sizes), H * W)
        _, k_windows_indices = torch.topk(router_compute.mean(2), k_window_size, dim=1)
        top_k_list = k_windows_indices.tolist()[0]
        
        top_k_window_sizes = [self.window_sizes[i] for i in top_k_list]
        

        processed_wins_attn_list = []

        for window_size in top_k_window_sizes:
            num_windows_h = H // window_size

            num_windows_w = W // window_size


            k_windows = int(self.k_window_rate * num_windows_h * num_windows_w)
            ego_windows = x[:, 0].view(B, C, num_windows_h, window_size, num_windows_w, window_size)
            ego_windows = ego_windows\
                .permute(0, 2, 4, 1, 3, 5) \
                .contiguous() \
                .view(B, num_windows_h * num_windows_w, C, window_size, window_size)
            other_windows = x[:, 1:]\
                .view(B, L - 1, C, num_windows_h, window_size, num_windows_w, window_size)
            other_windows = other_windows\
                .permute(0, 1, 3, 5, 2, 4, 6) \
                .contiguous() \
                .view(B, (L - 1), num_windows_h * num_windows_w, C, window_size, window_size)
            
            attn_module = self.attn_modules[str(window_size)]
            
            
            local_router_weights = attn_module.router(ego_windows.permute(0, 1, 3, 4, 2))
            topk_weights_value, topk_indices = torch\
                .topk(local_router_weights, int(k_windows), dim=1)
            selected_windows_ego = ego_windows\
                .gather(1, topk_indices\
                    .view(B, -1, 1, 1, 1) \
                        .expand(B, k_windows, C, window_size, window_size))

            selected_windows_other = other_windows.permute(1, 0, 2, 3, 4, 5) \
                .gather(2, topk_indices\
                    .view(1, B, -1, 1, 1, 1)\
                        .expand(L - 1, -1, -1, C, window_size, window_size)) \
                .permute(1, 0, 2, 3, 4, 5)
            attn_output_ego, attn_output_other = attn_module(selected_windows_ego, selected_windows_other)

            attn_output_ego = attn_output_ego * topk_weights_value\
                .unsqueeze(2).unsqueeze(3).unsqueeze(4)
            attn_output_other = attn_output_other * topk_weights_value\
                .unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(1)

            processed_attn_ego = torch.zeros_like(ego_windows) \
                .scatter_(1, topk_indices.view(B, -1, 1, 1, 1)\
                    .expand(-1, -1, C, window_size, window_size),
                          attn_output_ego)
            
            processed_attn_others = torch.zeros_like(other_windows)\
                .permute(1, 0, 2, 3, 4, 5) \
                    .scatter_(2, topk_indices.view(1, B, -1, 1, 1, 1) \
                            .expand(L - 1, -1, -1, C, window_size, window_size),
                            attn_output_other.permute(1, 0, 2, 3, 4, 5)) \
                    .permute(1, 0, 2, 3, 4, 5)
            processed_attn_ego = processed_attn_ego\
                .view(B, C, H, W)\
                    .unsqueeze(1)
            processed_attn_others = processed_attn_others.view(B, L - 1, C, H, W)

            processed_wins_attn_list.append(
                torch.concat((processed_attn_ego, processed_attn_others), dim=1)\
                    .permute(0, 1, 3, 4, 2))
            

        if self.window_fusion_method == 'split':
            x = self.window_fusion(processed_wins_attn_list)
        else:
            x = sum(processed_wins_attn_list) / len(processed_wins_attn_list)

        return  x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, window_size, window_num):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size

        self.to_ego_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_other_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)

        # self.router = nn.Linear(window_num, window_num)
        self.router = ExpertRouter(window_size=window_size, expert_mid_dim=128, feature_dim=dim, num_experts=8)

        self.to_out_ego = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.to_out_other = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, ego_feats, other_feats):
        B, k_windows, C, window_size, window_size = ego_feats.shape
        B, L, k_windows, C, window_size, window_size = other_feats.shape

        qkv_ego = self.to_ego_qkv(ego_feats.permute(0, 1, 3, 4, 2).unsqueeze(1))\
            .chunk(3, dim=-1)
        # (B, L, k_windows, window_size, window_size, C * 3)
        qkv_others = self.to_other_qkv(other_feats.permute(0, 1, 2, 4, 5, 3))\
            .chunk(3, dim=-1)
        q_ego, k_ego, v_ego = map(
            lambda t: rearrange(t,
                                'b l k s w (m c) -> b m k (s w) l c',
                                k=k_windows, s=window_size,
                                m=self.heads, w=window_size), qkv_ego)
        
        q_others, k_others, v_others = map(
            lambda t: rearrange(t,
                                'b l k s w (m c) -> b m k (s w) l c',
                                k=k_windows, s=window_size,
                                m=self.heads, w=window_size), qkv_others)
        
        attn_ego = torch.einsum('bmknic,bmknjc->bmknij', q_ego, k_others) * self.scale
        # (B, heads, k_windows, (window_size*window_size), L, 1)
        attn_other = torch.einsum('bmknic,bmknjc->bmknij', q_others, k_ego) * self.scale

        attn_ego = attn_ego.softmax(dim=-1)
        attn_other = attn_other.softmax(dim=-1)
        # (B, heads, k_windows, (window_size*window_size), l = 1, c)
        out_ego = torch.einsum('bmknij,bmknjc->bmknic', attn_ego, v_others)
        # (B, heads, k_windows, (window_size*window_size), l, c)
        out_other = torch.einsum('bmknij,bmknjc->bmknic', attn_other, v_ego)

        out_ego = rearrange(out_ego,
                            'b m k (s w) l c -> b l k s w (m c)',
                            k=k_windows, s=window_size,
                            m=self.heads, w=window_size)
        out_other = rearrange(out_other,
                              'b m k (s w) l c -> b l k s w (m c)',
                              k=k_windows, s=window_size,
                              m=self.heads, w=window_size)

        # out_ego.shape: (B, k_windows, C, H, W)
        out_ego = self.to_out_ego(out_ego).permute(0, 1, 2, 5, 3, 4).squeeze(1)
        out_other = self.to_out_other(out_other).permute(0, 1, 2, 5, 3, 4)

        #  out_ego.shape
        #  out_other.shape
        return out_ego, out_other

class WindowFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU() 

    def forward(self, x):
        
        x = torch.stack(x, dim=2) 
        B, L, N, C, H, W = x.shape
        
        x = torch.mean(x, dim=2).view(-1, C, H, W)  
        # x2.shape:{x.shape}
        x = self.conv(x)  
        # x3.shape:{x.shape}
        x = x.view(B, L, C, H, W)
        # x4.shape:{x.shape}
        return x


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(batch,
                       cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=3)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
       
        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        B, L = sw.shape[0], sw.shape[1]
        x_gap = sw + mw + bw
        
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)

        out = sw * x_attn[:, :, :, :, 0:self.input_dim] + \
              mw * x_attn[:, :, :, :, self.input_dim:2*self.input_dim] +\
              bw * x_attn[:, :, :, :, self.input_dim*2:]

        return out



class ExpertRouter(nn.Module):
    def __init__(self, window_size, feature_dim, expert_mid_dim, num_experts):
        super(ExpertRouter, self).__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim

        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(window_size * window_size * feature_dim, expert_mid_dim),
            nn.ReLU(),
            nn.Linear(expert_mid_dim, 1)
        ) for _ in range(num_experts)])
 
        self.gating_net = nn.Linear(window_size * window_size * feature_dim, num_experts)

    def forward(self, x):
        batch_size, num_windows, window_size, window_size, feature_dim = x.size()
        x = x.reshape(batch_size * num_windows, window_size * window_size * feature_dim)
        
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  
        gating_weights = torch.softmax(self.gating_net(x), dim=-1)  
        weighted_outputs = gating_weights.unsqueeze(-1) * expert_outputs 
        
        final_output = torch.sum(weighted_outputs, dim=1).view(batch_size, num_windows) 
        final_output = torch.sigmoid(final_output)
        return final_output

