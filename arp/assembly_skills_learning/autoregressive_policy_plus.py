from math import ceil
from copy import deepcopy
import wandb
from collections import defaultdict, ChainMap
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from typing import Optional, Tuple
import torch

import torchvision
import numpy as np
import clip
from torch.cuda.amp import autocast, GradScaler
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.optim import Lamb, GradualWarmupScheduler
from utils.structure import ActResult
from argparse import Namespace
import utils.math3d as math3d
from utils.clip import clip_encode_text
from PIL import Image
from preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
    flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts, \
    apply_se3_augmentation, transform_pc, grid_sample_from_heatmap, add_uniform_noise, denorm_rgb

from utils.layers import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding
)

from arp.util.arp import AutoRegressivePolicy, TokenType, LayerType, ModelConfig
from autoregressive_policy import MultiViewTransformer, Policy
import math

class PolicyNetwork(nn.Module):
    def __init__(self, model_cfg, env_cfg, render_device):
        super().__init__()
        self._num_rotation_classes = model_cfg.num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._image_resolution = [env_cfg.image_size, env_cfg.image_size]
        self._transform_augmentation = model_cfg.transform_augmentation
        self._place_with_mean = model_cfg.place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(np.array(model_cfg.transform_augmentation_xyz))
        self._transform_augmentation_rpy = model_cfg.transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = self._rotation_resolution

        self.gt_hm_sigma = model_cfg.gt_hm_sigma
        self.add_rgc_loss = model_cfg.add_rgc_loss
        self.amp = model_cfg.amp

        self.scene_bounds = env_cfg.scene_bounds
        self.cameras = env_cfg.cameras
        self.move_pc_in_bound = model_cfg.move_pc_in_bound

        self.rotation_aug = model_cfg.rotation_aug # 2
        self.stage2_zoom_scale = model_cfg.stage2_zoom_scale # st_sca
        self.stage2_waypoint_label_noise = model_cfg.stage2_waypoint_label_noise # st_wpt_loc_aug
        self.point_augment_noise = model_cfg.point_augment_noise # img_aug_2

        self.num_all_rot = self._num_rotation_classes * 3
        self.proprio_dim = model_cfg.proprio_dim
        self.img_size = model_cfg.img_size
        self.img_patch_size = model_cfg.img_patch_size
        self.renderer = CubePointCloudRenderer(render_device, (model_cfg.img_size, model_cfg.img_size), with_depth=model_cfg.add_depth, cameras=model_cfg.mvt_cameras)
        self.num_cameras = len(model_cfg.mvt_cameras)
        if model_cfg.render_with_cpp:
            assert model_cfg.mvt_cameras == ['top', 'left', 'front']
            self.render_with_cpp = True
            from point_renderer.rvt_renderer import RVTBoxRenderer
            self.cpp_renderer = RVTBoxRenderer(device=render_device,
                                               img_size=(model_cfg.img_size, model_cfg.img_size),
                                               three_views=True,
                                               with_depth=model_cfg.add_depth)
        else:
            self.render_with_cpp = False

        self.mvt1 = MultiViewTransformer(model_cfg, renderer=self.renderer)
        self.mvt2 = MultiViewTransformer(model_cfg, renderer=self.renderer)

        self.spatial_logits_buffer = []

        def sample_callback(lst_of_spatial_logits):
            assert len(lst_of_spatial_logits) == 1
            self.spatial_logits_buffer.append(lst_of_spatial_logits[0])
            bs = len(lst_of_spatial_logits[0])
            dev = lst_of_spatial_logits[0].device
            return torch.zeros(bs, 1, 2, device=dev) # dummy output

        self.sample_callback = sample_callback

        # produce each xyz for stage 1
        # then use xyz feature as a condition, to produce each xyz for stage 2
        # then produce rot and grip separately

        #region parse moe properties
        self.is_transformer_moe = model_cfg.is_transformer_moe if hasattr(model_cfg, "is_transformer_moe") else False
        self.is_emb_moe = model_cfg.is_emb_moe if hasattr(model_cfg, "is_emb_moe") else False
        self.moe_weight = model_cfg.moe_weight if hasattr(model_cfg, "is_transformer_moe") and hasattr(model_cfg, "is_emb_moe") else None
        self.moe_multiple_gate = model_cfg.moe_multiple_gate if hasattr(model_cfg, "moe_multiple_gate") else False
        self.moe_cfg = model_cfg.moe if hasattr(model_cfg, "moe") else None
        #endregion
        
        #region parse moe properties
        if hasattr(model_cfg, "rot_z_weight_factor"): # compatible with the old config
            self.use_exponential_weight_flag: bool = True
            self.rot_z_weight_factor: float = model_cfg.rot_z_weight_factor
            self.rot_z_weight_max: float = model_cfg.rot_z_weight_max
            self.exponential_weight: float = self.rot_z_weight_max
        else:
            self.use_exponential_weight_flag: bool = False
            self.exponential_weight: float = 1.0
        self.step:int = 0
        #endregion
        
        arp_cfg = ModelConfig(
            n_embd=128,
            embd_pdrop = 0.1, 
            max_seq_len = 6 + 6 + 3 + 2,
            max_chunk_size = 2, # grip and collision
            layer_norm_every_block=False,
            tokens=[
                TokenType.make(name='prompt-features', dim=1, 
                            embedding='discrete', is_control=True, 
                            embedding_kwargs={'embed_from': "prompt-features"}), 

                TokenType.make(
                        name='stage1-screen-pts', dim=2, is_continuous=True, dict_sizes=[self.img_size, self.img_size],
                        embedding="zero", predictor="upsample_from_2d_attn", 
                        predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': self.img_patch_size, 'label_name': 'smooth-heatmap'}),
                TokenType.make(
                    name='stage2-screen-pts', dim=2, is_continuous=True, dict_sizes=[self.img_size, self.img_size],
                    embedding="zero", predictor="upsample_from_2d_attn", 
                    predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': self.img_patch_size, 'label_name': 'smooth-heatmap'}), 
            ] +  [
                TokenType.make(name=f'rot-{c}', dim=1, is_continuous=False, dict_sizes=[self._num_rotation_classes], embedding='position_1d', predictor='class', predictor_kwargs={'label_name': f'rot-{c}'}) for c in ['x', 'y', 'z']
            ] + [
                TokenType.make(name='grip', dim=1, is_continuous=False, dict_sizes=[2], embedding='discrete', predictor='class'),
                TokenType.make(name='collision', dim=1, is_continuous=False, dict_sizes=[2], embedding='discrete', predictor='class')
            ],
            layers=[
                LayerType.make(n_head=8, AdaLN=True, condition_on='visual-tokens', name='cross')
            ] * 4 + [
                LayerType.make(n_head=8, name='self')
            ] * 6,
            
            is_transformer_moe=self.is_transformer_moe,
            is_emb_moe=self.is_emb_moe,
            moe_multiple_gate=self.moe_multiple_gate,
            moe_cfg=self.moe_cfg,            
        )
        self.policy = AutoRegressivePolicy(arp_cfg)
        
        # gripper state only depends on xyz, but not rotation
        self.block_attn_directions = [(n, f'rot-{c}') for c in ['x', 'y', 'z'] for n in ['grip', 'collision']]
        self.cfg = model_cfg

    def update(self):
        self.step += 1
        if self.use_exponential_weight_flag:
           self.update_weight_exponential()
    
    def update_weight_exponential(self):
        """Linear Decay Weight
            Gradually reduce the weight over time:

            𝑤(𝑡)=1+(max_weight−1)⋅exp(-alpha*𝑡)
            
            t: self.step - the time step for the weight to decay
            max_weight⋅: self.rot_z_weight_max - the maximum value of W(t)
            alpha: self.rot_z_weight_factor - how fast the weight decays to one
        """        
        self.exponential_weight = 1 + (self.rot_z_weight_max - 1) * math.exp(-self.rot_z_weight_factor * self.step)
    
    def multi_view_coordinate_sampler(self, lst_of_spatial_logits):
        hm_logits = torch.cat([a for a in lst_of_spatial_logits], dim=1)
        hm = F.softmax(hm_logits.flatten(2), dim=2)
        bs = len(hm_logits)
        hm = hm.view(bs, 3, 224, 224)
        pred_pt = [self.renderer.get_most_likely_point_3d(hm[i : i + 1]) for i in range(bs)]
        spatial_point = torch.cat(pred_pt, 0) # bs, 3
        screen_points = self.renderer.points3d_to_screen2d(spatial_point[:, None, :])
        screen_points = screen_points[:, 0]
        return spatial_point, screen_points
    
    def to_tk_reg_ids(self, token_name_regs):
        result = []
        for v in token_name_regs:
            r = [self.token_name_2_ids[v[0]], v[1]]
            if len(v) == 3: r.append(v[2])
            result.append(r)
        return result

    def get_gt_rot_grip_collision(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """
        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = math3d.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision (to one hot, if result = 0, then don't ignore collision)
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_gt_translation_action(
        self,
        waypoint, # this is groundtruth 3d point
        dims,
    ):
        """
        Generate action_trans and wpt_img
            action_trans: normalized gaussian heatmap generated from wp_img, the heatmap is centered on wpt_img, [bs, h*w, 3(nc)]
            wpt_img: 2D locations (x, y) on rendered images mapped from 3d waypoints (x, y, z), shape (bs, nc, 2)
        """ 
        # note: will be called separately for stage 1 / 2
        bs, nc, h, w = dims
        wpt_img = self.renderer.points3d_to_screen2d(waypoint.unsqueeze(1)) # (bs, np, num_cameras, 2) # torch.Size([48, 3, 2])
        assert wpt_img.shape[1] == 1
        wpt_img = wpt_img.squeeze(1)  # (bs, num_img, 2)
        action_trans = generate_heatmap_from_screen_pts(
            wpt_img.reshape(-1, 2), #! just the winning points
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()
        return action_trans, wpt_img

    def render(self, pc, img_feat, mvt: MultiViewTransformer):
        """Render point cloud pc and image feature im_feat to images
        """        
        # renderer(inputs) returns images of shape  [num_images, height, width, channels]
        renderer = self.cpp_renderer if self.render_with_cpp else self.renderer
        with torch.no_grad():
            with autocast(enabled=False):
                if mvt.add_corr:
                    # Correlate pc and image feature
                    if mvt.norm_corr:
                        # normalize pc with max pc
                        img = []
                        for _pc, _img_feat in zip(pc, img_feat):
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            img.append(
                                renderer(_pc, torch.cat(((_pc / max_pc), _img_feat), dim=-1)).unsqueeze(0) # [3, 224, 224, 7], 3 -> views, 7 -> feats
                            )
                    else:
                        # correlate pc and image feature but do not normalize pc
                        img = [renderer(_pc, torch.cat((_pc, _img_feat), dim=-1)).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]
                else:
                    # render image from pc and image feature
                    img = [renderer(_pc, _img_feat).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]

        # Stack the images then permute dimension to [batch_size, num_views, channels, height, width]
        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]

        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device) # extra feature
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )
        return img

    def forward(self, observation):
        loss_dicts = []
        nc, h, w = len(self.cfg.mvt_cameras), self.img_size, self.img_size
        dev = observation["lang_goal_embs"].device
        if self.training:
            action_grip = observation["gripper_action"].int() # (b,) of int
            action_ignore_collisions = observation["ignore_collisions"].view(-1, 1).int()  # (b, 1) of int
            action_gripper_pose = observation["gripper_pose"]  # (b, 7)
            action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3), translation in xyz
            action_rot = action_gripper_pose[:, 3:7]  # (b, 4), rotation in quaternion xyzw

        lang_goal_embs = observation["lang_goal_embs"].float()
        proprio = observation["low_dim_state"]

        #region preprocess and augmentation

        obs, pcd = preprocess_images_in_batch(observation, self.cameras)
        pc, img_feat = flatten_img_pc_to_points(obs, pcd)

        
        with torch.no_grad():
            if self._transform_augmentation and self.training:
                action_trans_con, action_rot, pc = apply_se3_augmentation( #! where the gt really comes out (for SE3 trans)
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
                action_rot = action_rot.cpu().numpy()
                for i, _action_rot in enumerate(action_rot):
                    _action_rot = math3d.normalize_quaternion(_action_rot)
                    if _action_rot[-1] < 0:
                        _action_rot = -_action_rot
                    action_rot[i] = _action_rot
        pc, img_feat = clamp_pc_in_bound(pc, img_feat, self.scene_bounds, skip=not self.move_pc_in_bound)
        pc_new, rev_trans_stage1, waypoint_stage1 = [], [], []

        # Prepare point clouds for Stage 1 by centering them within a cube
        for i, _pc in enumerate(pc):
            a, b = place_pc_in_cube(_pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            if self.training:
                waypoint_stage1.append(place_pc_in_cube(_pc, action_trans_con[i][:3],
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0].unsqueeze(0))
            pc_new.append(a)
            rev_trans_stage1.append(b)
        # 缩放后的pc
        pc = pc_new
        bs = len(pc)

        # Combine waypoints and apply noise augmentation during training
        if self.training:
            waypoint_stage1 = torch.cat(waypoint_stage1, axis=0).clone().detach()
            if self.point_augment_noise != 0:
                with torch.no_grad():
                    for x in img_feat:
                        stdv = self.point_augment_noise * torch.rand(1, device=x.device)
                        noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                        x += noise

        # Render images with gt pc and noisy img_feat for Stage 1
        img = self.render(pc, img_feat, self.mvt1)# img shape [batch_size, num_views, channels, height, width]
        #endregion ###########################

        #  extracts visual feature maps
        visual_featmap_1 = self.mvt1(img=img, proprio=proprio, lang_emb=lang_goal_embs) # [B, num_cameras, 128, np, np]

        #region Stage 1
        if self.training:
            # Generate ground truth spatial heatmap  # smooth_spatial_label_stage1 = action_trans.shape([48, 50176, 3]); screen_waypoint_stage1 = wpt_img.shape(([48, 3, 2]))
            smooth_spatial_label_stage1, screen_waypoint_stage1 = self.get_gt_translation_action(waypoint_stage1, dims=(bs, nc, h, w))
            stage1_chk_ids = torch.as_tensor([0], device=dev)[None, :]

            # the 0, 0 are dummy input
            # 这里的seq[0,0,2]
            seq = torch.as_tensor([0, 0, self.policy.token_name_2_ids['stage1-screen-pts']], device=dev).reshape(1, 1, 3).repeat(bs, 1, 1)
            
            # Compute loss for each camera view 
            # between noisy visual feature map visual_featmap_1 (this featmap generated from noisy img_feat)
            # and ground truth smooth_spatial_label_stage1 
            tmp_loss_dict = defaultdict(list)
            for view_id in range(3):
                _loss_dict = self.policy.compute_loss(seq, stage1_chk_ids, match_layer='cross',
                            contexts={
                                'visual-tokens': visual_featmap_1[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                'visual-featmap': visual_featmap_1[:, view_id],
                                'smooth-heatmap': smooth_spatial_label_stage1[:, :, view_id]
                            },
                            task_ids = observation['task_idx'])
                for k, v in _loss_dict.items():
                    tmp_loss_dict[k].append(v)
            
            # Combine losses for Stage 1 with mean
            loss_dicts.append({k: sum(v) / len(v) for k, v in tmp_loss_dict.items()})
        else:
            # Inference: Generate waypoint predictions for Stage 1
            prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage1-screen-pts'])]

            assert len(self.spatial_logits_buffer) == 0
            for view_id in range(3):
                self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
                                    contexts={
                                            'visual-tokens': visual_featmap_1[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                            'visual-featmap': visual_featmap_1[:, view_id],
                                    },
                                    task_ids = observation['task_idx']
                                    )
                assert len(self.spatial_logits_buffer) == (view_id + 1)
            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in self.spatial_logits_buffer], dim=1)
            
            # predict the 3d point in a cube
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage1 = torch.cat(pred_pt, 0) # bs, 3
            self.spatial_logits_buffer.clear()
        #endregion

        #region Stage 2
        with torch.no_grad():
            if self.training:
                # Add uniform noise on groundtruth waypoint 
                # and transform it to point clouds
                waypoint_stage1_noisy = add_uniform_noise(
                    waypoint_stage1.clone().detach(), 2 * self.stage2_waypoint_label_noise
                )
                # 输出的pc是加噪变形后的pc: point-wise将pc - waypoint_stage1_noisy，然后再按照self.stage2_zoom_scale放大
                # rev_trans_stage2是能把加噪变形后的pc还原的函数
                pc, rev_trans_stage2 = transform_pc(pc, loc=waypoint_stage1_noisy, sca=self.stage2_zoom_scale)
                # waypoint_stage2：加噪变形后的waypoint
                # 把gt轨迹减去waypoint_stage1_noisy,并且放大到self.stage2_zoom_scale
                waypoint_stage2, _ = transform_pc(waypoint_stage1, loc=waypoint_stage1_noisy, sca=self.stage2_zoom_scale)
            else:
                # Transform the way point predicted from stage1
                pc, rev_trans_stage2 = transform_pc(pc, loc=waypoint_stage1, sca=self.stage2_zoom_scale)
                waypoint_stage1_noisy = waypoint_stage1
                waypoint_stage2 = None
        # Render visual features for Stage 2. Same img_feat but with mvt2
        img = self.render(pc, img_feat, self.mvt2)
        visual_featmap_2 = self.mvt2(img=img, proprio=proprio, lang_emb=lang_goal_embs)

        if self.training:
            (
                action_rot_x,
                action_rot_y,
                action_rot_z,
                action_grip,       # (bs)
                action_collision,  # (bs)
            ) = [v.argmax(-1) for v in self.get_gt_rot_grip_collision(bs, action_rot, action_grip, action_ignore_collisions, device=dev)]
            
            # Add rotation noise if needed
            if self.rotation_aug:
                rotation_aug = torch.from_numpy(np.random.choice(self.rotation_aug[0], p=self.rotation_aug[1], size=(bs, 3))).to(dev)
                action_rot_aug_x = action_rot_x + rotation_aug[:, 0]
                action_rot_aug_y = action_rot_y + rotation_aug[:, 1]
                action_rot_aug_z = action_rot_z + rotation_aug[:, 2]
            else:
                action_rot_aug_x = action_rot_x
                action_rot_aug_y = action_rot_y
                action_rot_aug_z = action_rot_z
            action_rot_aug_x %= self._num_rotation_classes
            action_rot_aug_y %= self._num_rotation_classes
            action_rot_aug_z %= self._num_rotation_classes
            smooth_spatial_label_stage2, screen_waypoint_stage2 = self.get_gt_translation_action(waypoint_stage2, dims=(bs, nc, h, w))

            stage2_chk_ids = torch.as_tensor([0], device=dev)[None, :]
            seq = torch.as_tensor([0, 0, self.policy.token_name_2_ids['stage2-screen-pts']], device=dev).reshape(1, 1, 3).repeat(bs, 1, 1)
            tmp_loss_dict = defaultdict(list)
            
            # compute cross entropy loss for visual feature map2
            for view_id in range(3):
                _loss_dict = self.policy.compute_loss(seq, stage2_chk_ids, match_layer='cross',
                            contexts={
                                'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                'visual-featmap': visual_featmap_2[:, view_id],
                                'smooth-heatmap': smooth_spatial_label_stage2[:, :, view_id]
                            },
                            task_ids=observation['task_idx'])
                for k, v in _loss_dict.items():
                    tmp_loss_dict[k].append(v)
            loss_dicts.append({k: sum(v) / len(v) for k, v in tmp_loss_dict.items()})

            # ------------------------------------------- #
            
            # # TODO　多尺度特征融合强化特征提取
            # B, V, C, H, W = visual_featmap_2.shape

            # # 1) 先把不同尺度特征做 1×1 降到同一通道数，例如 D=128
            # D = C  
            # lateral_convs = []
            # for v in range(V):
            #     # 给每个 view 一套 FPN lateral conv
            #     lateral_convs.append(nn.Conv2d(C, D, kernel_size=1).to(visual_featmap_2.device))

            # # 2) top‐down pathway + 融合
            # fpn_feats = []  # 存每一层融合后的特征
            # for view_id in range(V):
            #     # 拿到这一路
            #     feats = visual_featmap_2[:, view_id]   # [B, C, H, W]
            #     # bottom (level 0)
            #     c3 = lateral_convs[view_id](feats)     # [B, D, H, W]
            #     # 构造更小尺度：C4: 1/2、C5: 1/4
            #     c4 = F.max_pool2d(c3, kernel_size=2)    # [B, D, H/2, W/2]
            #     c5 = F.max_pool2d(c4, kernel_size=2)    # [B, D, H/4, W/4]

            #     # top‐down：把 C5 上采样 + C4 做融合 → P4
            #     p5 = c5
            #     p4 = F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False) + c4
            #     p3 = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False) + c3

            #     # 全局池化：P3,P4,P5 每个做 GAP → [B, D]
            #     for p in (p3, p4, p5):
            #         pooled = p.flatten(2).max(-1)[0]  # torch.max over spatial → [B, D]
            #         fpn_feats.append(pooled)

            # # 3) 将所有 view 和所有尺度拼到一起
            # # fpn_feats 总共有 V * 3 个 [B, D]
            # prompt_features = torch.stack(fpn_feats, dim=1)
            # # → shape [B, V*3, D]

            # compute gripper loss
            prompt_features = torch.cat([ # [bs, 6, 128]
                    grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
                                            visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
                    visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            P = prompt_features.shape[1]
            
            seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(P)], 
                                  device=dev).reshape(1, P, 2).repeat(bs, 1, 1)
            seq = torch.cat([seq, torch.cat([
                                    torch.cat([
                                        action_rot_aug_x[:, None, None],
                                        action_rot_aug_y[:, None, None],
                                        action_rot_aug_z[:, None, None],
                                        action_grip[:, None, None],
                                        action_collision[:, None, None]], dim=1), 
                                    torch.as_tensor([self.policy.token_name_2_ids[k] for k in ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision']], 
                                                    device=dev)[None, :, None].repeat(bs, 1, 1)], dim=-1)
                             ], dim=1)
            chk_ids = torch.as_tensor(list(range(5 + P)), device=dev)[None, :] # 11表示：Stage 2 需要同时预测“rot-x, rot-y, rot-z, grip, collision”（5 个 token）再加上前面的 6（多尺度是９） 个 prompt features，共 11 个位置，导致 chunk length=11。
            loss_dict_gripper = self.policy.compute_loss(seq, chk_ids,
                                                         block_attn_directions=self.block_attn_directions, 
                                                         match_layer='self', contexts={
                                                            'prompt-features': prompt_features,
                                                            'rot-x': action_rot_x[:, None],
                                                            'rot-y': action_rot_y[:, None], 
                                                            'rot-z': action_rot_z[:, None]
                                                         },
                                                         task_ids=observation['task_idx'])
            loss_dicts.append(loss_dict_gripper)
        else:
            # Generate future waypoint from prompt
            prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage2-screen-pts'])]
            for view_id in range(3):
                self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
                                    contexts={
                                            'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                            'visual-featmap': visual_featmap_2[:, view_id],
                                    },
                                    task_ids=observation['task_idx'])
                assert len(self.spatial_logits_buffer) == (view_id + 1)

            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in self.spatial_logits_buffer], dim=1)
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage2 = torch.cat(pred_pt, 0) # bs, 3
            self.spatial_logits_buffer.clear()

            screen_waypoint_stage2 = self.renderer.points3d_to_screen2d(waypoint_stage2[:, None, :])[:, 0]

            prompt_features = torch.cat([ # [bs, 6, 128]
                    grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
                                            visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
                    visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            P = prompt_features.shape[1]
            
            prompt_seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(P)], 
                                  device=dev).reshape(1, P, 2).repeat(bs, 1, 1)
            future_tk_chk_ids = [dict(chk_id=chk_id, tk_id=self.policy.token_name_2_ids[tk_name]) 
                                 for chk_id, tk_name in zip(range(P, 5 + P), ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision'])]
            
            result_seq_stage2 = self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='self', 
                                                    sample=False,  block_attn_directions=self.block_attn_directions,  
                                                    contexts={
                                                        'prompt-features': prompt_features
                                                    },
                                                    task_ids=observation['task_idx'])
        #endregion
        
        #region final output
        # Get loss dictionary when training, get action when inferencing
        if self.training:
            loss_dict = {}
            for d in loss_dicts: loss_dict.update(d)
            norm = lambda x: torch.norm(x.flatten(1), dim=1).mean().item()
            loss_dict['stat_dict'] = {
                    'v1_norm': norm(visual_featmap_1.flatten(0, 1)),
                    'v2_norm': norm(visual_featmap_2.flatten(0, 1))
            }
            # normalized with the number of elements in aux loss tensor
            loss_dict['aux_loss'] = loss_dict['aux_loss'].sum()/loss_dict['aux_loss'].numel() * self.moe_weight
            if 'aux_loss_adapter' in loss_dict:
                loss_dict['aux_loss_adapter'] = loss_dict['aux_loss_adapter'].sum()/loss_dict['aux_loss_adapter'].numel() * self.moe_weight
            
            return loss_dict
        else:
            final_waypoint = rev_trans_stage1[0](rev_trans_stage2(waypoint_stage2))
            pred_rot = result_seq_stage2[:, 6:9, 0]
            pred_rot_quat = math3d.discrete_euler_to_quaternion(pred_rot.cpu().numpy(), self._rotation_resolution)
            continuous_action = np.concatenate(
                (
                    final_waypoint[0].cpu().numpy(),
                    pred_rot_quat[0],
                    result_seq_stage2[:, 9, 0].cpu().numpy(),
                    result_seq_stage2[:, 10, 0].cpu().numpy(),
                )
            )
            return continuous_action
        #endregion
        
