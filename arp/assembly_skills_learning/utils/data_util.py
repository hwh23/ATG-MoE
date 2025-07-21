from typing import Any, List
import os.path as osp
import numpy as np
from rlbench.demo import Demo
from PIL import Image
from pyrep.objects import VisionSensor
from utils.clip import clip_encode_text
import clip
from utils.structure import *
from rlbench.backend.const import DEPTH_SCALE
from scipy.spatial.transform import Rotation as R
import OpenEXR
import Imath
import torch
import os
import torch
import open3d as o3d
from .logging_config import get_logger

logger = get_logger(__name__, False)

##region process utils
def get_gripper_joint_positions(obs:dict):
    """Returns a 2-element float32 np.ndarray representing gripper joint positions."""
    if obs.get('gripper_joint_positions') is not None:
        gripper_joint_positions = obs('gripper_joint_positions')
    elif obs.get('gripper_open') is not None:
        gripper_open_factor = 0.04
        gripper_joint_positions = [obs['gripper_open'] * gripper_open_factor] * 2
    else:
        gripper_joint_positions = [0.0, 0.0]
    return np.array(gripper_joint_positions, dtype=np.float32)  # 2

def get_reasonable_low_dim_state(obs:dict): # dim=18
    low_dim_state = []
    if obs.get('gripper_open') is not None:
        low_dim_state.append(float(obs['gripper_open'] > 0.5))  # 1
    if obs.get('ignore_collisions') is not None:
        low_dim_state.append(obs['ignore_collisions'])
    if obs.get('gripper_joint_positions') is not None:
        low_dim_state.extend(obs['gripper_joint_positions'])  # 2
    if obs.get('joint_positions') is not None:
        low_dim_state.extend(obs['joint_positions'])
    if obs.get('gripper_pose') is not None:
        low_dim_state.extend(obs['gripper_pose'])
    return np.array(low_dim_state, dtype=np.float32) # 18

def get_low_dim_state(origin_style_state, obs:dict):
    if origin_style_state:
        curr_low_dim_state = np.array([float(obs['gripper_open'] > 0.5) ,
                                        *get_gripper_joint_positions(obs)
                                        ])
    else:
        curr_low_dim_state = get_reasonable_low_dim_state(obs)
    return torch.tensor(curr_low_dim_state, dtype=torch.float32).to('cpu')

def get_lang_emb(obs, add_lang:bool, debug_mode, clip_model, device):
    """
    Extracts or generates language goal embeddings from the observation dictionary.

    Depending on the configuration and the contents of `obs`, this method will:
    - Use existing language goal embeddings if present.
    - Encode language goal tokens or description text using a CLIP model if available.
    - Otherwise, generate zero embeddings of the appropriate shape.

    Args:
        obs (dict): Observation dictionary that may contain language goal information such as
            'lang_goal_embs', 'lang_goal_tokens', or 'description'.

    Returns:
        torch.Tensor: The language goal embeddings, stored in `obs['lang_goal_embs']`.
    """
    if add_lang:
        if 'lang_goal_embs' in obs:
            pass
        elif debug_mode:
            lang_goal_embs = torch.zeros((77, 512), dtype=torch.float32)
        elif 'lang_goal_tokens' in obs:
            _, lang_goal_embs = clip_encode_text(clip_model, lang_goal_tokens)
        elif 'description' in obs:
            lang_goal_tokens = clip.tokenize(obs['description'])
            lang_goal_tokens = lang_goal_tokens.to(device) 
            _, lang_goal_embs = clip_encode_text(clip_model, lang_goal_tokens)
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = torch.zeros((77, 512), dtype=torch.float32)
    else:
        lang_goal_embs = torch.zeros((77, 512), dtype=torch.float32)
    obs['lang_goal_embs'] = lang_goal_embs.to('cpu')
    return obs['lang_goal_embs']

def append_rgbd_pc_to_obs(msg:dict, obs:dict, visualize:bool=False)->None:
    logger.debug('append rgb pc')
    for camera, cur_view in msg['images'].items():
        logger.debug(f'Processing {camera}')
        rgb = torch.tensor([])
        depth = torch.tensor([])
        point_cloud = torch.tensor([])
    
        rgb_shape = cur_view['rgb'].shape
        d_shape = cur_view['depth'].shape
        cur_view['rgb'] = torch.tensor(cur_view['rgb'], dtype=torch.float32).to('cpu')
        
        # Dimension verification
        if len(rgb_shape) == 3:
            # Turn rgb from (CHW) to (BCHW)
            cur_view['rgb'] = cur_view['rgb'].unsqueeze(0)
        if rgb_shape[1] == max(rgb_shape):
            # Turn rgb from (B,H,W,C) to (B,C,H,W)
            cur_view['rgb'] = cur_view['rgb'].permute(0, 3, 1, 2)
        if rgb_shape[1] == 4:
            # Convert rgba to rgb
            a = np.asarray(cur_view['rgb'][0,3,:,:], dtype='float32' ) / 255.0
            cur_view['rgb'][0,0,:,:] = cur_view['rgb'][0,0,:,:] * a + (1.0 - a) * 255
            cur_view['rgb'][0,1,:,:] = cur_view['rgb'][0,1,:,:] * a + (1.0 - a) * 255
            cur_view['rgb'][0,2,:,:] = cur_view['rgb'][0,2,:,:] * a + (1.0 - a) * 255
            cur_view['rgb'] = np.delete(cur_view['rgb'], 3, axis=1)
        if len(d_shape) == 3:
            # Turn depth from (1,H,W) to (H,W)
            cur_view['depth'] = cur_view['depth'].squeeze(0)
            
        # Image processing: output RGB at shape [(bsz)1, (channel)3, H, W]; D at shape [H, W]； pc at shape [B,C,H,W]
        rgb = cur_view['rgb']
        logger.debug(f'[Process] {camera}_rgb shape: {rgb.shape}')
        
        # Denormalize depth before getting the actual depth metres
        depth_np = cur_view['depth']*10
        
        logger.info(f'[Process] {camera} intrinsic: {np.array(msg["misc"][camera]["intrinsics"])}')
        T_cw_cv, T_cv_cr = unity_extvec_to_extrinsics(np.array(msg['misc'][camera]['extrinsics']))
                                                                              
        point_cloud_np = pointcloud_from_depth_and_camera_params( # it takes depth at shape (H,W)
                                                                np.array(depth_np),
                                                                T_cw_cv,
                                                                np.array(
                                                                    msg['misc'][camera]['intrinsics']
                                                                    ).reshape(3,3)
                                                                )
        if visualize: save_combined_pointcloud_and_gripper({f'{camera}_point_cloud': point_cloud_np, 'gripper_pose': msg['gripper_pose']}, 
                                                           camera=camera, save_dir='data/train/visualize_pc_tcp/cv')
        point_cloud_np = transform_pointcloud_world_to_robot(point_cloud_np, T_cv_cr) # transform point cloud to robot base frame
        if visualize: save_combined_pointcloud_and_gripper({f'{camera}_point_cloud': point_cloud_np, 'gripper_pose': msg['gripper_pose']}, 
                                                            camera=camera, save_dir='data/train/visualize_pc_tcp/world')
        
        if visualize: save_depth_to_exr(depth_map=depth_np, path=f'data/train/visualize_pc_tcp/{camera}_depth.exr')
        
        point_cloud = torch.tensor(point_cloud_np, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to('cpu') # turn point cloud to shape [B,C,H,W]
        logger.debug(f'[Process] {camera}_pointcloud shape: {point_cloud.shape}')
        
        depth = torch.tensor(depth_np, dtype=torch.float32).to('cpu')
        logger.debug(f'[Process] {camera}_depth shape: {depth.shape}')
        
        obs[f'{camera}_rgb'] = rgb
        obs[f'{camera}_depth'] = depth
        obs[f'{camera}_point_cloud'] = point_cloud
##endregion

def save_depth_to_exr(depth_map, path):
    # depth_map: H x W, float32 numpy array
    height, width = depth_map.shape
    header = OpenEXR.Header(width, height)

    # Set channel as float32 (HALF for float16 if desired)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

    exr = OpenEXR.OutputFile(path, header)
    
    # OpenEXR expects bytes per channel (as string), so convert
    depth_bytes = depth_map.astype(np.float32).tobytes()

    # Save depth in R channel
    exr.writePixels({'R': depth_bytes})
    exr.close()


def read_exr(exr_path):
    """读取EXR文件并返回深度数据"""
    try:
        # 打开EXR文件
        exr_file = OpenEXR.InputFile(exr_path)
        
        # 获取图像尺寸
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        # 读取深度通道
        # 注意：Unity Perception生成的深度图通道名可能是'Z'或'R'，根据实际情况调整
        try:
            depth_str = exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
        except:
            depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
        
        # 将字节数据转换为numpy数组
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(size[1], size[0])
        
        return depth
    except Exception as e:
        logger.error(f"读取EXR文件 {exr_path} 时出错: {e}")
        return None

def quaternion_pose_to_axis_angle(pose_quat: np.ndarray) -> np.ndarray:
    """
    Convert 6-DoF pose from quaternion format to axis-angle format.

    Args:
        pose_quat (np.ndarray): shape (7,) or (N,7).  
            Each row is [tx, ty, tz, qx, qy, qz, qw]
            where (qx, qy, qz, qw) is the quaternion (scalar last format).

    Returns:
        np.ndarray: shape (6,) or (N,6).  
            Each row [tx, ty, tz, rx, ry, rz], 
            where (rx, ry, rz) is the axis-angle rotation vector.
    """
    # --- Handle single pose case ---
    single = False
    if pose_quat.ndim == 1:
        pose_quat = pose_quat[np.newaxis, :]  # (1,7)
        single = True
    elif pose_quat.ndim != 2 or pose_quat.shape[1] != 7:
        raise ValueError(f"pose_quat must be shape (7,) or (N,7), got {pose_quat.shape}")

    # Split translation and quaternion
    translation = pose_quat[:, :3]  # (N,3)
    quat_xyzw = pose_quat[:, 3:]   # (N,4)

    # Quaternion (x,y,z,w) -> axis-angle
    rotation = R.from_quat(quat_xyzw)
    axis_angle = rotation.as_rotvec()  # (N,3)

    # Combine back [t, axis_angle]
    pose_aa = np.concatenate([translation, axis_angle], axis=1)  # (N,6)

    # If input was 1D, return 1D
    if single:
        return pose_aa[0]
    return pose_aa

# 轴角转换为四元数
def axis_angle_to_quaternion_pose(pose_aa: np.ndarray) -> np.ndarray:
    """
    Convert 6-DoF pose from axis-angle format to quaternion format.

    Args:
        pose_aa (np.ndarray): shape (6,) or (N,6).  
            每行 [tx, ty, tz, rx, ry, rz], 
            其中 (rx, ry, rz) 是 axis-angle 旋转向量。

    Returns:
        np.ndarray: shape (7,) or (N,7).  
            每行 [tx, ty, tz, qx, qy, qz, qw]
    """
    # --- 兼容单个 pose ---
    single = False
    if pose_aa.ndim == 1:
        pose_aa = pose_aa[np.newaxis, :]  # (1,6)
        single = True
    elif pose_aa.ndim != 2 or pose_aa.shape[1] != 6:
        raise ValueError(f"pose_aa must be shape (6,) or (N,6), got {pose_aa.shape}")

    # split translation and axis-angle
    translation = pose_aa[:, :3]  # (N,3)
    axis_angle  = pose_aa[:, 3:]  # (N,3)

    # axis-angle -> quaternion (x,y,z,w)
    rotation = R.from_rotvec(axis_angle)
    quat_xyzw = rotation.as_quat()  # (N,4)

    # 拼回 [t, quat]
    pose_quat = np.concatenate([translation, quat_xyzw], axis=1)  # (N,7)

    # 如果原来是 1D，返回 1D
    if single:
        return pose_quat[0]
    return pose_quat

# 四元数向旋转矩阵的转换函数
def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    q: array_like, shape (4,), in order [qx, qy, qz, qw]
    returns: R, shape (3,3)
    """
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    # quaternion to rotation matrix
    R = np.array([
        [1 - 2 * (y*y + z*z),     2 * (x*y - z*w),         2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x*x + z*z),         2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)]
    ])
    return R

def oned_ext_to_trans_mat(ext_vec: np.ndarray) -> np.ndarray:
    """
    Convert extrinsics vector to 4x4 transformation matrix.

    Parameters:
      ext_vec: shape (7,), [tx, ty, tz, qx, qy, qz, qw]
        translation and quaternion in world frame.

    Returns:
      ext_mat: shape (4,4), extrinsics matrix.
    """
    assert ext_vec.shape == (7,)
    # Extract translation and quaternion
    t = ext_vec[:3]
    q = ext_vec[3:7]

    # Convert quaternion to rotation matrix
    R = quat_to_rot_matrix(q)

    # Build 4x4 transformation matrix
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = R
    ext_mat[:3, 3] = t
    return ext_mat

# reshape相机外参矩阵
def cam_ext_world(ext_vec: np.ndarray) -> np.ndarray:
    """
    Convert camera extrinsics from world LH (Y-up) to robot RH (Z-up).

    Parameters:
      ext_vec: shape (7,), [tx, ty, tz, qx, qy, qz, qw]
        translation and quaternion in world frame.

    Returns:
      ext_mat_robot: shape (4,4), extrinsics matrix in robot frame.
    """
    assert ext_vec.shape == (7,)
    # Extract
    t_w = ext_vec[:3]
    q_w = ext_vec[3:7]

    # Convert quaternion to rotation in world frame
    R_w = quat_to_rot_matrix(q_w)

    # Build 4x4 extrinsics in robot frame
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = R_w
    ext_mat[:3, 3]  = t_w
    return ext_mat

# 将相机坐标从世界坐标系下向机械臂坐标系下转换（采集到的数据，同坐标原点，左右手坐标系不同）
def cam_ext_world_to_robot(ext_vec: np.ndarray) -> np.ndarray:
    """
    Convert camera extrinsics from world LH (Y-up) to robot RH (Z-up).

    Parameters:
      ext_vec: shape (7,), [tx, ty, tz, qx, qy, qz, qw]
        translation and quaternion in world frame.

    Returns:
      ext_mat_robot: shape (4,4), extrinsics matrix in robot frame.
    """
    assert ext_vec.shape == (7,)
    # Extract
    t_w = ext_vec[:3]
    q_w = ext_vec[3:7]

    # Convert quaternion to rotation in world frame
    R_w = quat_to_rot_matrix(q_w)

    # Define mapping from world LH (Y-up) to robot RH (Z-up)
    M = np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [ 0,  1,  0]
    ])

    # Transform rotation and translation
    R_r = M @ R_w @ M.T # M有修改，同时R也需要×M-1？
    t_r = M @ t_w

    # Build 4x4 extrinsics in robot frame
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = R_r
    ext_mat[:3, 3]  = t_r
    return ext_mat

# 如果给的EXR是ndc归一化的深度图，则需要将其转换为实际的深度值
def linearize_depth_ndc(d_ndc, near, far):
    # Unity/D3D 规范化公式
    return (near * far) / (far - d_ndc * (far - near))

def unity_extvec_to_extrinsics(ext_vec: np.ndarray,cam_to_world: bool = True):
    """
    输入:
      ext_vec: shape (7,) = [tx, ty, tz, qx, qy, qz, qw]
        — Unity 世界坐标系下的相机位置 + 四元数
        — Unity 用左手系 (Y-up, Z→里)
    返回:
      T_cw_cv: (4×4) Camera→OpenCV-World 外参，
                用于 pointcloud_from_depth_and_camera_params
      T_cv_r:  (4×4) OpenCV-World → Robot-Base 坐标映射，
                用于将上述 world_pc 转到机械臂坐标系下
    """

    # 拆分平移与四元数
    t_u = ext_vec[:3]
    q_u = ext_vec[3:]            # Unity 四元数 (x,y,z,w)

    # Unity 四元数 → 3×3 旋转（左手系）
    R_u = R.from_quat(q_u).as_matrix()

    # 2) 如果输入的 ext_vec 是 World→Camera，先取逆变成 Camera→World
    if not cam_to_world:
        R_u = R_u.T
        t_u = -R_u @ t_u

    # Unity-World (LH Y↑ Z→) → OpenCV-World (RH X→ Y↓ Z→)
    M_u2cv = np.array([
        [ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
    ], dtype=float)

    # 构造 Camera→OpenCV-World 外参
    R_cv = M_u2cv @ R_u @ M_u2cv.T
    t_cv = M_u2cv @ t_u
    T_cw_cv = np.eye(4, dtype=float)
    T_cw_cv[:3,:3] = R_cv
    T_cw_cv[:3, 3] = t_cv

    # OpenCV-World (RH X→ Y↓ Z→里) → Robot-Base (RH X→  Y→里  Z↑) 
    # rotate 90 deg clockwise around x-axis
    M_cv2r = np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [ 0, -1,  0],
    ], dtype=float)

    # Rotate 90 deg anti clockwise around X-axis
    # M_cv2r = np.array([
    #     [1, 0,  0],
    #     [0, 0, -1],
    #     [0, 1,  0]
    # ], dtype=float)

    # 齐次矩阵
    T_cv_r = np.eye(4, dtype=float)
    T_cv_r[:3,:3] = M_cv2r

    return T_cw_cv, T_cv_r

def transform_pointcloud_world_to_robot(
        world_pc: np.ndarray,
        T_wr: np.ndarray) -> np.ndarray:
    """
    Transform a point cloud from world coordinate system to robot coordinate system.

    Parameters:
      world_pc: np.ndarray of shape (..., 3)
        Point cloud in world coordinates. Can be (H, W, 3) or (N, 3).
      T_wr: np.ndarray of shape (4, 4)
        Homogeneous transformation matrix from world to robot frame.

    Returns:
      robot_pc: np.ndarray of same shape as world_pc
        Point cloud in robot coordinates.
    """
    # Flatten points to shape (3, N)
    original_shape = world_pc.shape
    pts = world_pc.reshape(-1, 3).T  # (3, N)

    # Extract rotation and translation
    R_wr = T_wr[:3, :3]  # (3, 3)
    t_wr = T_wr[:3, 3].reshape(3, 1)  # (3, 1)

    # Apply transformation: p_r = R_wr * p_w + t_wr
    pts_r = R_wr @ pts + t_wr  # (3, N)

    # Reshape back to original
    robot_pc = pts_r.T.reshape(original_shape)
    return robot_pc

# 写个新的，代替VisionSensor.pointcloud_from_depth_and_camera_params
def pointcloud_from_depth_and_camera_params(depth: np.ndarray,
                                            extrinsics: np.ndarray,
                                            intrinsics: np.ndarray
                                            ) -> np.ndarray:
    """
    Converts depth (in meters) to a point cloud in the world frame,
    using explicit inv(K) and R,t, rather than inverting a big projection matrix.

    Inputs:
      depth:       (H, W) 深度图 in meters
      extrinsics:  (4,4) Camera→World 齐次矩阵
      intrinsics:  (3,3) 相机内参矩阵 K

    Returns:
      pc_w:        (H, W, 3) point cloud in world frame
    """
    H, W = depth.shape
    # 1) 构建归一化像素坐标 [u,v,1] 
    us = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)  # (H, W)
    vs = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)  # (H, W)
    # 但真实的成像模型里，像素坐标原点是「像素中心」而不是「像素左上角」  
    # us = (np.arange(W, dtype=np.float32) + 0.5)[None, :].repeat(H, 0)
    # vs = (np.arange(H, dtype=np.float32) + 0.5)[:, None].repeat(W, 1)

    ones = np.ones_like(us)                                    # (H, W)
    pix = np.stack((us, vs, ones), axis=0).reshape(3, -1)      # (3, H*W)

    # 2) 反投影到相机系：dirs = inv(K) @ pix, pts_cam = dirs * depth
    invK = np.linalg.inv(intrinsics)
    dirs_cam = invK @ pix                    # (3, H*W)
    pts_cam = dirs_cam * depth.reshape(1, -1)  # (3, H*W)

    # 3) 相机系 → 世界系：pts_w = R @ pts_cam + t
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3:4]                 # (3,1)

    pts_w = R @ pts_cam + t                  # (3, H*W)

    # 4) 恢复形状 (H, W, 3)
    pc_w = pts_w.T.reshape(H, W, 3)

    # 增加相机坐标系下的点云输出，有需要可以return
    pc_cam= pts_cam.T.reshape(H, W, 3)  # (H, W, 3)
    return pc_w

# # 如果深度不是光轴深度而是欧式距离，用这个函数处理
def pointcloud_from_depth_and_camera_params_euclidean_distance(depth: np.ndarray,
                                            extrinsics: np.ndarray,
                                            intrinsics: np.ndarray) -> np.ndarray:
    """
    Converts depth (in meters) to a point cloud in the world frame,
    using explicit inv(K) and R,t, rather than inverting a big projection matrix.

    Inputs:
      depth:       (H, W) 深度图 in meters
      extrinsics:  (4,4) Camera→World 齐次矩阵
      intrinsics:  (3,3) 相机内参矩阵 K

    Returns:
      pc_w:        (H, W, 3) point cloud in world frame
    """

    H, W = depth.shape
    # 1) 像素坐标 u,v
    us = (np.arange(W) + 0.5)[None, :].repeat(H, 0)  # 半像素中心
    vs = (np.arange(H) + 0.5)[:, None].repeat(W, 1)
    pix = np.stack((us, vs, np.ones_like(us)), axis=0).reshape(3, -1)

    # 2) 反投影方向向量
    invK = np.linalg.inv(intrinsics)
    dirs_cam = invK @ pix                  # (3, H*W), [x_norm, y_norm, 1]

    # ←—— 在这里插入：把 depth_raw（欧氏距离）还原成 z_c（光轴深度） ——→
    norms = np.sqrt(dirs_cam[0]**2 + dirs_cam[1]**2 + 1.0)      # (H*W,)
    z_c   = depth.reshape(-1) / norms                           # (H*W,)
    # ←——————————————————————————————————————————————————————————————→

    # 3) 用真正的 z_c 做相机系坐标
    pts_cam = dirs_cam * z_c[np.newaxis, :]                     # (3, H*W)

    # 4) 相机系 → 世界系
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3:4]
    pts_w = R @ pts_cam + t                                      # (3, H*W)

    # 5) reshape 并返回
    return pts_w.T.reshape(H, W, 3)


def retreive_full_observation(cameras, essential_obs, episode_path, i, load_mask=False, skip_rgb=False):
    
    IMAGE_RGB = 'rgb'
    IMAGE_DEPTH = 'depth'
    RGB_FORMAT  = '%d.png'
    DEPTH_FORMAT = '%d.exr'  # 使用EXR格式读取深度图

    obs = {}
  
    # 超参数DEPTH_SCALE和深度图编码关系有关，八位深度图编码为0-255，十六位深度图编码为0-65535
    for camera in cameras:
        if load_mask:
            obs[f"{camera}_mask"] = np.array(
                Image.open(osp.join(episode_path, f"{camera}_depth", RGB_FORMAT % i))# mask 改成depth
            )
        if not skip_rgb:
            obs[f"{camera}_rgb"] =  np.array(Image.open(osp.join(episode_path, '%s_%s' % (camera, IMAGE_RGB), RGB_FORMAT % i)).convert("RGB"))
        # obs[f'{camera}_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        obs[f'{camera}_depth'] = read_exr(osp.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), DEPTH_FORMAT % i)) # 返回H*W的numpy数组
        # ============
        # near = essential_obs.misc['%s_camera_near' % (camera)]
        # far = essential_obs.misc['%s_camera_far' % (camera)]
        # obs[f'{camera}_depth'] = near + obs[f'{camera}_depth'] * (far - near)
        # ============
        T_cw_cv, T_cv_cr = unity_extvec_to_extrinsics(np.array(essential_obs.misc[f'{camera}_camera_extrinsics']))
        obs[f'{camera}_point_cloud'] = pointcloud_from_depth_and_camera_params_euclidean_distance(obs[f'{camera}_depth'],
                                                                                            T_cw_cv,
                                                                                            np.array(essential_obs.misc[f'{camera}_camera_intrinsics']).reshape(3,3)
                                                                                            )
        obs[f'{camera}_point_cloud'] = transform_pointcloud_world_to_robot(obs[f'{camera}_point_cloud'], T_cv_cr) 
    return obs

def print_pointcloud_xyz_ranges(obs, cameras):
    logger.info("\n--- 相机点云坐标范围统计 ---")
    for camera in cameras:
        key = f"{camera}_point_cloud"
        if key not in obs:
            logger.info(f"[{camera}] 没有点云数据")
            continue

        pc = obs[key]  # shape: (H, W, 3)
        pc_flat = pc.reshape(-1, 3)
        valid_mask = ~np.isnan(pc_flat).any(axis=1) & ~np.isinf(pc_flat).any(axis=1)
        pc_valid = pc_flat[valid_mask]

        if pc_valid.size == 0:
            logger.info(f"[{camera}] 无有效点云点")
            continue

        x_min, y_min, z_min = pc_valid.min(axis=0)
        x_max, y_max, z_max = pc_valid.max(axis=0)

        logger.info(f"[{camera}]")
        logger.info(f"  X: ({x_min:.3f}, {x_max:.3f})")
        logger.info(f"  Y: ({y_min:.3f}, {y_max:.3f})")
        logger.info(f"  Z: ({z_min:.3f}, {z_max:.3f})")
    logger.info("------------------------------\n")

def encode_time(t, episode_length=25):
    return (1. - (t / float(episode_length - 1))) * 2. - 1.

def quat_angle_diff(q0: np.ndarray, q1: np.ndarray) -> float:
    """
    Compute the smallest rotation angle (in radians) between two unit quaternions q0, q1.
    Quaternions are given in [x, y, z, w] order.

    Args:
        q0, q1: arrays of shape (4,), unit quaternions.

    Returns:
        angle (float): the rotation angle (radians) between q0 and q1.
    """
    # dot product
    dot = np.dot(q0, q1)
    # q and -q represent the same rotation: take absolute value
    dot = np.clip(abs(dot), -1.0, 1.0)
    # angle
    return 2.0 * np.arccos(dot)


def _is_stopped(
    demo,
    i: int,
    stopped_buffer: int,
    pos_thresh: float = 1e-4,
    rot_thresh: float = 1e-3,
    joint_pos_thresh: float = 1e-4
) -> bool:
    """
    Determine whether the demonstration at index i is 'stopped' (no motion).

    Args:
        demo: sequence of step objects, each having attributes:
            - gripper_pose: array-like shape (7,) [tx, ty, tz, qx, qy, qz, qw]
            - gripper_open: scalar (e.g., 0 or 1)
            - joint_positions: array-like of joint values
        i: current index in the demo sequence
        stopped_buffer: int, buffer counter (must be <=0 to consider stopping)
        pos_thresh: translation threshold (meters)
        rot_thresh: rotation threshold (radians)
        joint_pos_thresh: joint position threshold

    Returns:
        True if stopped, False otherwise.
    """
    # need at least one frame before and after
    if i <= 0 or i >= len(demo) - 1:
        return False

    # previous, current, next
    prev_step = demo[i - 1]
    curr_step = demo[i]
    next_step = demo[i + 1]

    # --- translation check ---
    p0 = np.array(prev_step.gripper_pose[:3])
    p1 = np.array(curr_step.gripper_pose[:3])
    p2 = np.array(next_step.gripper_pose[:3])
    pos_delta = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)
    pos_unchanged = pos_delta < pos_thresh

    # --- rotation check via quaternion angle diff ---
    q0 = np.array(prev_step.gripper_pose[3:7])
    q1 = np.array(curr_step.gripper_pose[3:7])
    q2 = np.array(next_step.gripper_pose[3:7])
    angle01 = quat_angle_diff(q0, q1)
    angle12 = quat_angle_diff(q1, q2)
    rot_unchanged = (angle01 + angle12) < rot_thresh
    #   代替原先的处理，更贴近四元数空间
    #   rot_delta = np.linalg.norm(quat1 - quat0) + np.linalg.norm(quat2 - quat1)
    #   rot_unchanged = rot_delta < rot_thresh

    # --- gripper open/close check ---
    g0 = prev_step.gripper_open
    g1 = curr_step.gripper_open
    g2 = next_step.gripper_open
    # unchanged if there is no difference at all
    gripper_unchanged = (g0 == g1 == g2)

    # --- joint position check ---
    j0 = np.array(prev_step.joint_positions)
    j1 = np.array(curr_step.joint_positions)
    j2 = np.array(next_step.joint_positions)
    joint_delta = np.linalg.norm(j1 - j0) + np.linalg.norm(j2 - j1)
    joint_unchanged = joint_delta < joint_pos_thresh

    # final decision
    return (stopped_buffer <= 0
            and pos_unchanged
            and rot_unchanged
            and gripper_unchanged
            and joint_unchanged)

""" RLBench code has a different version of _is_stopped, which is not used here.
def _is_stopped(demo, i, stopped_buffer, pos_thresh=1e-4, rot_thresh=1e-3, joint_pos_thresh=1e-4):
    sec_next_is_not_final = i <= (len(demo) - 2)
    if sec_next_is_not_final is False:
        return False
    
    pos0 = demo[i-1].gripper_pose[:3]
    pos1 = demo[i].gripper_pose[:3]
    pos2 = demo[i+1].gripper_pose[:3]
    
    quat0 = demo[i-1].gripper_pose[3:]
    quat1 = demo[i].gripper_pose[3:]
    quat2 = demo[i+1].gripper_pose[3:]
    
    gripper0 = demo[i-1].gripper_open
    gripper1 = demo[i].gripper_open
    gripper2 = demo[i+1].gripper_open
    
    pos_delta = np.linalg.norm(pos1 - pos0) + np.linalg.norm(pos2 - pos1)
    pos_unchanged = pos_delta < pos_thresh
    
    rot_delta = np.linalg.norm(quat1 - quat0) + np.linalg.norm(quat2 - quat1)
    rot_unchanged = rot_delta < rot_thresh
    
    gripper_delta = abs(gripper1 - gripper0) + abs(gripper2 - gripper1) + abs(gripper2 - gripper0) # 0 for unchanged, > 1 for changed
    gripper_unchanged = gripper_delta == 0
    
    joint_pos_delta = np.linalg.norm(demo[i].joint_positions - demo[i-1].joint_positions) + \
                          np.linalg.norm(demo[i+1].joint_positions - demo[i].joint_positions)
    joint_pos_unchanged = joint_pos_delta < joint_pos_thresh 
    
    stopped = (stopped_buffer <= 0 and pos_unchanged and rot_unchanged and gripper_unchanged and joint_pos_unchanged)
    return stopped

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i != (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
            obs.gripper_open == demo[i - 1].gripper_open and
            demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
            next_is_not_final and gripper_state_no_change)
    return stopped
"""

def keypoint_discovery(demo: Demo, stopping_delta: float=0.1, stop_buffer_max = 4) -> List[int]:
    episode_keypoints = []
    # prev_gripper_status = get_gripper_status(demo[0].gripper_open)
    prev_gripper_status = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        # stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped = _is_stopped(demo, i, stopped_buffer) # check if the gripper is stationary
        stopped_buffer = stop_buffer_max if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        # current_gripper_status = get_gripper_status(obs.gripper_open)
        current_gripper_status = obs.gripper_open
        if i != 0 and (current_gripper_status != prev_gripper_status or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_status = current_gripper_status
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    return episode_keypoints


def query_next_kf(f, kfs, return_index=False):
    for i, kf in enumerate(kfs):
        if kf > f:
            if return_index:
                return i
            else:
                return kf
    raise RuntimeError("No more keyframes")


def get_reasonable_low_dim_state(essential_obs): # dim=18
    return np.array([
            essential_obs.gripper_open,
            essential_obs.ignore_collisions,
            *essential_obs.gripper_joint_positions,
            *essential_obs.joint_positions,
            *essential_obs.gripper_pose
        ]).astype(np.float32) # 18

# 点云可视化
def save_combined_pointcloud_and_gripper(sample_dict, camera, save_dir, axis_sampling_steps=50):
    """
    将点云和 gripper 坐标系一起保存到一个 PLY 文件（点形式）：
      - {save_dir}/{camera}_combined.ply
    通过在坐标轴线上采样离散点并赋予颜色，实现"同文件同视图"展示。
    :param axis_sampling_steps: 在每条轴线段上采样多少个点
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) 原始点云
    pc = sample_dict[f"{camera}_point_cloud"]  # H x W x 3
    # pc = sample_dict[f"{camera}_point_cloud_cam_raw"]  # H x W x 3

    pts = pc.reshape(-1, 3)

    # 点云着色为灰色
    pc_colors = np.tile(np.array([0.5, 0.5, 0.5]), (pts.shape[0], 1))

    # 2) 构造 gripper 坐标系轴线上的离散采样点及颜色
    gp = sample_dict['gripper_pose']  # [tx, ty, tz, qx, qy, qz, qw]
    t = gp[:3]
    x, y, z, w = gp[3:]
    # 四元数 -> 旋转矩阵
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    length = 0.05
    axis_pts = []
    axis_colors = []
    # 红、绿、蓝分别对应 X, Y, Z 轴
    axes_colors = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    for idx, col in enumerate(axes_colors):
        start = t
        end = t + R[:, idx] * length
        # 在 start->end 上等间隔采样
        for alpha in np.linspace(0, 1, axis_sampling_steps):
            pt = start * (1 - alpha) + end * alpha
            axis_pts.append(pt)
            axis_colors.append(col)
    axis_pts = np.array(axis_pts)
    axis_colors = np.array(axis_colors)

    # 3) 合并点及颜色
    all_pts = np.vstack([pts, axis_pts])
    all_colors = np.vstack([pc_colors, axis_colors])

    # 4) 构建并写出组合点云
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_pts)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    out_path = os.path.join(save_dir, f"{camera}_combined.ply")
    o3d.io.write_point_cloud(out_path, combined_pcd, write_ascii=True)
    logger.info(f"[{camera}] saved combined PLY (with colored axes) to {out_path}")