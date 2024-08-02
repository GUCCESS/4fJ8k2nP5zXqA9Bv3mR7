import torch
import torch.nn as nn 

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.rocooperfuse import *
from opencood.models.fuse_modules.fuse_utils import regroup



class PointPillarERRCO(nn.Module):
    
    def __init__(self, args):
        super(PointPillarERRCO, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # self.loss_model = CQI_loss_model(config=args['loss_model'])
        self.fusion_net = RoCooperFusion(args=args['RoCooper'])
        
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()


    def forward(self, data_dict):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        voxel_features = data_dict['processed_lidar']['voxel_features'].to(device)
        voxel_coords = data_dict['processed_lidar']['voxel_coords'].to(device)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'].to(device)
        record_len = data_dict['record_len'].to(device)
        spatial_correction_matrix = data_dict['spatial_correction_matrix'].to(device)
        
        meta_data =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']


        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        regroup_feature, mask = regroup(spatial_features_2d,  
                                        record_len,
                                        self.max_cav)
        meta_data = meta_data.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4]).to(device)
        
        regroup_feature = torch.cat([regroup_feature, meta_data], dim=2)
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        
        
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
      

        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        output_dict = {'psm': psm,
                       'rm': rm}
        return output_dict
    
    
    
    
    
    
    def backbone_fix(self):

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    
    