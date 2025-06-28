import clip
import torch
import random
import numpy.random as npr
from typing import Any, List
from tqdm import tqdm
import os.path as osp
import numpy as np
import os
import json
import logging
from rlbench.demo import Demo
import pickle
from PIL import Image
from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor
from dataclasses import dataclass
from collections import defaultdict
import utils.math3d as math3d
from utils.clip import clip_encode_text
from rlbench.backend.observation import Observation
from utils.structure import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from rlbench.backend.const import DEPTH_SCALE
from scipy.spatial.transform import Rotation as R

# TODO　初步修改完成，常量的定义在utils.structure
# # def get_demo_essential_info(data_path, episode_ind):
# def get_proprio_info(data_path, episode_ind):
#     EPISODE_FOLDER = 'episode%d'
#     episode_path = osp.join(data_path, EPISODE_FOLDER % episode_ind)
#     # low dim pickle file
#     with open(osp.join(episode_path, PROPRIOCEPTION), 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return {
#             'end_pose': data.get('position'),
#             'joint': data.get('joint'),
#             'time_step': data.get('step')
#         }

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
        [ 0,  0, -1],
        [ 0,  1,  0]
    ])

    # Transform rotation and translation
    R_r = M @ R_w
    t_r = M @ t_w

    # Build 4x4 extrinsics in robot frame
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = R_r
    ext_mat[:3, 3]  = t_r
    return ext_mat

def retreive_full_observation(cameras, essential_obs, episode_path, i, load_mask=False, skip_rgb=False):
    
    IMAGE_RGB = 'rgb'
    IMAGE_DEPTH = 'depth'
    IMAGE_FORMAT  = '%d.png'

    obs = {}
  
    # TODO：确认一下这个超参数DEPTH_SCALE
    for camera in cameras:
        if load_mask:
            obs[f"{camera}_mask"] = np.array(
                Image.open(osp.join(episode_path, f"{camera}_depth", IMAGE_FORMAT % i))# mask 改成depth
            )
        if not skip_rgb:
            obs[f"{camera}_rgb"] =  np.array(Image.open(osp.join(episode_path, '%s_%s' % (camera, IMAGE_RGB), IMAGE_FORMAT % i)).convert("RGB"))
        obs[f'{camera}_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = essential_obs.misc['%s_camera_near' % (camera)]
        far = essential_obs.misc['%s_camera_far' % (camera)]
        obs[f'{camera}_depth'] = near + obs[f'{camera}_depth'] * (far - near)
        obs[f'{camera}_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs[f'{camera}_depth'],
                                                                                            cam_ext_world_to_robot(np.array(essential_obs.misc[f'{camera}_camera_extrinsics'])),
                                                                                            np.array(essential_obs.misc[f'{camera}_camera_intrinsics']).reshape(3,3)
                                                                                            )
    return obs


def encode_time(t, episode_length=25):
    return (1. - (t / float(episode_length - 1))) * 2. - 1.

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
    
    gripper0 = demo[i-1].gripper_pose[-1]
    gripper1 = demo[i].gripper_pose[-1]
    gripper2 = demo[i+1].gripper_pose[-1]
    
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

# def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
#     next_is_not_final = i != (len(demo) - 2)
#     gripper_state_no_change = (
#             i < (len(demo) - 2) and
#             (obs.gripper_open == demo[i + 1].gripper_open and
#             obs.gripper_open == demo[i - 1].gripper_open and
#             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
#     small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
#     stopped = (stopped_buffer <= 0 and small_delta and
#             next_is_not_final and gripper_state_no_change)
#     return stopped

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


class TransitionDataset(Dataset):
    def __init__(self, root: str, tasks: List[str], cameras:List[str]=["front", "left_shoulder", "right_shoulder", "wrist"],
                batch_num: int=1000, batch_size: int=6, scene_bounds=[-0.3,-0.5,0.6,0.7,0.5,1.6],
                voxel_size:int=100, rotation_resolution:int=5, cached_data_path=None,
                origin_style_state=True,
                episode_length=25, time_in_state=False, k2k_sample_ratios={}, o2k_window_size=10,
                shuffle:bool=False):
        super().__init__()
        self._num_batches = batch_num
        self._batch_size = batch_size
        self.tasks = tasks
        self.cameras = cameras
        self.origin_style_state = origin_style_state
        self.shuffle=shuffle
        if not origin_style_state:
            assert not time_in_state, "should not include a discrete timestep in state"

        self.episode_length = episode_length
        self.root = root
        self.k2k_sample_ratios = k2k_sample_ratios
        self.o2k_window_size = o2k_window_size

        self.scene_bounds = scene_bounds
        self.voxel_size = voxel_size
        self.rotation_resolution = rotation_resolution
        self.include_time_in_state = time_in_state

        # Load the CLIP model (ViT-B/32 is common)
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        model, preprocess = clip.load("ViT-B/32", device=device)

        # task -> episode_id -> step_id
        if cached_data_path and osp.exists(cached_data_path):
            self.data = torch.load(cached_data_path)
        else:
            self.data = {}
            for task in tqdm(tasks, desc="building meta data"):
                episodes_path = osp.join(root, task, 'all_variations/episodes')
                if task not in self.data: self.data[task] = {}
                for episode in tqdm(os.listdir(episodes_path), desc="episodes", leave=False):
                    if 'episode' not in episode:
                        continue
                    else:
                        if episode not in self.data[task]: self.data[task][episode] = dict(keypoints=[], lang_emb=None, obs=None)
                        ep = osp.join(episodes_path, episode)
                        self.load_lowdim_obs_variation_num(ep, task, episode)
                        self.load_keypoints(ep, task, episode)
                        self.load_lang_goal_emb(ep, task, episode, model, device)

            if cached_data_path:
                if not osp.exists(osp.dirname(cached_data_path)):
                    os.makedirs(osp.dirname(cached_data_path))
                torch.save(self.data, cached_data_path)
    
    def load_low_dim_obs(self, episode_folder: str) -> Demo:
        """
        Create a Demo object from the given path and episode ID.
        """
        proprioception_files = os.listdir(os.path.join(episode_folder, 'proprioception'))
        misc_file = os.path.join(episode_folder, CAMERA_JSON) 
        cameras = self.cameras
        observations = [0]*len(proprioception_files)
        with open(misc_file, 'r', encoding='utf-8') as f:
            misc = json.load(f)
            for cam in cameras:
                misc.update({# TODO 这里的相机参数需要和实际的相机名字对应
                    f"{cam}_camera_near": 0.1,
                    f"{cam}_camera_far": 100.0, 
                })
        
        for proprioception_file in proprioception_files:
            with open(os.path.join(episode_folder, 'proprioception', proprioception_file), 'r', encoding='utf-8') as f:
                proprioception = json.load(f)    
            
            gripper_open_continuous = np.array(proprioception.get('gripper_joint_positions', [0.0]*7)[-1])
            observation = Assembly_Observation(
                                    gripper_pose=np.array(proprioception.get('position', [0.0]*7)),    
                                    gripper_matrix=None,
                                    gripper_open=float(gripper_open_continuous > 0.5),
                                    gripper_joint_positions= np.repeat(gripper_open_continuous * 0.04, 2),
                                    gripper_touch_forces= None, #np.array(proprioception.get('gripper_touch_forces', [0.0]*6)),
                                    joint_positions=np.array(proprioception.get('joint', [0.0]*6)),
                                    joint_velocities=None,
                                    joint_forces=None,
                                    ignore_collisions=np.array(0.),
                                    task_low_dim_state= None, #np.array([0.0]*185),
                                    misc=misc
                                    )
            observations[proprioception['step']] = observation
        demo = Demo(observations=observations)
        return demo

    
    def load_lowdim_obs_variation_num(self, ep, task, episode):
        obs = []
        # Load low dimentional observations
        try:
            with open(osp.join(ep, LOW_DIM_PICKLE), 'rb') as f:
                obs = pickle.load(f)
        except FileNotFoundError:
            obs = self.load_low_dim_obs(ep)
        # load variation number
        try:
            with open(osp.join(ep, VARIATION_NUMBER_PICKLE), 'rb') as f:
                obs.variation_number = pickle.load(f)
        except FileNotFoundError:
            # TODO generate a random variation number currently
            obs.variation_number = 0
            with open(osp.join(ep, VARIATION_NUMBER_PICKLE), 'wb') as f:
                pickle.dump(obs.variation_number, f)
        self.data[task][episode]['obs'] = obs
            
    def load_keypoints(self, ep, task, episode):
        try: 
            with open(osp.join(ep, KEYPOINT_JSON)) as f:
                self.data[task][episode]['keypoints'] = json.load(f)
        except FileNotFoundError:
            logging.info(f"Keypoint file not found for {task} episode {episode}, generating keypoints.")
            kp:List = keypoint_discovery(self.data[task][episode]['obs'], 
                                         stopping_delta=0.1, 
                                         stop_buffer_max=20)
            self.data[task][episode]['keypoints'] = kp
            with open(osp.join(ep, KEYPOINT_JSON), 'w') as f:
                json.dump(kp, f)
    
    @torch.no_grad()
    def load_lang_goal_emb(self, ep, task, episode, model, device):
        try:
            with open(osp.join(ep, LANG_GOAL_EMB), 'rb') as f:
                self.data[task][episode]['lang_emb'] = pickle.load(f)
        except Exception as e:
            with open(osp.join(ep, DESC_PICKLE), 'r') as f:
                self.data[task][episode]['desc'] = json.load(f)
            tokens = clip.tokenize(self.data[task][episode]['desc']['instructions'][0]).to(device),
            tokens = tokens[0]
            _, embeddings = clip_encode_text(model, tokens)  # shape: [batch_size, embed_dim]
            # self.data[task][episode]['lang_tokens'] = tokens.cpu().numpy()
            self.data[task][episode]['lang_emb'] = embeddings.squeeze(0).cpu().numpy()
            with open(osp.join(ep, LANG_GOAL_EMB), 'wb') as f:
                pickle.dump(self.data[task][episode]['lang_emb'], f)    
            

    def __len__(self): return self._num_batches

    def get(self, **kwargs):
        return self.__getitem__(0, **kwargs)

    def __getitem__(self, _):
        batch = defaultdict(list)
        for _ in range(self._batch_size):
            task = random.choice(list(self.data.keys()))
            episode = random.choice(list(self.data[task].keys()))
            episode_idx = int(episode[len('episode'):])
            episode_path = osp.join(self.root, task, 'all_variations/episodes', episode)
            episode = self.data[task][episode]

            # --------------------------------------- #
            u = random.random()
            if u < self.k2k_sample_ratios.get(task, 0.8):
                # k2k
                kp = random.randint(0, len(episode['keypoints'])-1) #! target keypoint
                obs_frame_id = 0 if kp == 0 else episode['keypoints'][kp-1]
            else:
                # o2k
                obs_frame_id = episode['keypoints'][0]
                while obs_frame_id in episode['keypoints']:
                    obs_frame_id = random.randint(0, episode['keypoints'][-1])
                # obs_frame_id is just an ordinary frame, not key frame
                kp = query_next_kf(obs_frame_id, episode['keypoints'], return_index=True)

            # --------------------------------------- #

            kp_frame_id = episode['keypoints'][kp]
            variation_id = episode['obs'].variation_number
            essential_obs = episode['obs'][obs_frame_id]
            essential_kp_obs = episode['obs'][kp_frame_id]
            obs_media_dict = retreive_full_observation(self.cameras, essential_obs, episode_path, obs_frame_id)

            if self.origin_style_state:
                curr_low_dim_state = np.array([essential_obs.gripper_open, *essential_obs.gripper_joint_positions])
                if self.include_time_in_state:
                    curr_low_dim_state = np.concatenate(
                        [curr_low_dim_state,
                        [encode_time(kp, episode_length=self.episode_length)]]
                    ).astype(np.float32)
            else:
                curr_low_dim_state = get_reasonable_low_dim_state(essential_obs)

            sample_dict = {
                # "lang_goal_tokens": episode['lang_tokens'],
                "lang_goal_embs": episode['lang_emb'], 
                "keypoint_idx": kp,
                "kp_frame_idx": kp_frame_id,
                "frame_idx": obs_frame_id,
                "episode_idx": episode_idx,
                "variation_idx": variation_id,
                "task_idx": SKILL_TO_ID[task],

                "gripper_pose": essential_kp_obs.gripper_pose,
                "ignore_collisions": int(essential_kp_obs.ignore_collisions),

                "gripper_action": int(essential_kp_obs.gripper_open),
                "low_dim_state": curr_low_dim_state,

                **obs_media_dict
            }

            for k, v in sample_dict.items():
                batch[k].append(v)

            # reset
            task = episode = kp = obs_frame_id = None

        # lang_goals = batch.pop('lang_goals')
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = {k: torch.from_numpy(v.astype('float32') if v.dtype == np.float64 else v)
                for k, v in batch.items()}
        batch = {k: v.permute(0, 3, 1, 2) if k.endswith('rgb') or k.endswith('point_cloud')
                else v for k,v in batch.items()}
        # batch['lang_goals'] = lang_goals
        return batch


    def dataloader(self, num_workers=1, pin_memory=True, distributed=False, pin_memory_device=''):
        if distributed:
            sampler = DistributedSampler(self, shuffle=self.shuffle)
        else:
            sampler = RandomSampler(range(len(self)))
        if pin_memory and pin_memory_device != '':
            pin_memory_device = f'cuda:{pin_memory_device}'
        return DataLoader(self, batch_size=None, shuffle=False, pin_memory=pin_memory,
                        sampler=sampler, num_workers=num_workers, pin_memory_device=pin_memory_device), sampler

if __name__ == "__main__":
    only_key_frames_ratios = {
      "place_cups": 1,
      "stack_cups": 1,
      "close_jar": 1,
      "push_buttons": 1,
      "meat_off_grill": 1,
      "stack_blocks": 1,
      "reach_and_drag": 1,
      "slide_block_to_color_target": 1,
      "place_shape_in_shape_sorter": 1,
      "open_drawer": 1,
      "sweep_to_dustpan_of_size": 1,
      "put_groceries_in_cupboard": 1,
      "light_bulb_in": 1,
      "turn_tap": 1,
      "insert_onto_square_peg": 1,
      "put_item_in_drawer": 1,
      "put_money_in_safe": 1,
      "place_wine_at_rack_location": 1
    }
    D = TransitionDataset("./data/train", ["open_drawer"],
                          origin_style_state=True, time_in_state=False, k2k_sample_ratios=only_key_frames_ratios)
    D[0]


class Assembly_Observation(Observation):
    def __init__(self,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_velocities: np.ndarray = None,
                 
                 left_shoulder_rgb: np.ndarray = None,
                 left_shoulder_depth: np.ndarray = None,
                 left_shoulder_mask: np.ndarray = None,
                 left_shoulder_point_cloud: np.ndarray = None,
                 right_shoulder_rgb: np.ndarray = None,
                 right_shoulder_depth: np.ndarray = None,
                 right_shoulder_mask: np.ndarray = None,
                 right_shoulder_point_cloud: np.ndarray = None,
                 overhead_rgb: np.ndarray = None,
                 overhead_depth: np.ndarray = None,
                 overhead_mask: np.ndarray = None,
                 overhead_point_cloud: np.ndarray = None,
                 wrist_rgb: np.ndarray = None,
                 wrist_depth: np.ndarray = None,
                 wrist_mask: np.ndarray = None,
                 wrist_point_cloud: np.ndarray = None,
                 front_rgb: np.ndarray = None,
                 front_depth: np.ndarray = None,
                 front_mask: np.ndarray = None,
                 front_point_cloud: np.ndarray = None,
                 joint_forces: np.ndarray = None,
                 gripper_matrix: np.ndarray = None,
                 gripper_joint_positions: np.ndarray = None,
                 gripper_touch_forces: np.ndarray = None,
                 task_low_dim_state: np.ndarray = None,
                 ignore_collisions: np.ndarray = None,
                 misc :dict = None,
                 ):
        self.gripper_open: float = gripper_open
        self.gripper_pose: np.ndarray = gripper_pose
        self.joint_positions: np.ndarray = joint_positions
        self.joint_velocities: np.ndarray = joint_velocities
        super().__init__(
            gripper_open=gripper_open,
            gripper_pose=gripper_pose,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            # Cameras
            left_shoulder_rgb = left_shoulder_rgb,
            left_shoulder_depth = left_shoulder_depth,
            left_shoulder_mask = left_shoulder_mask,
            left_shoulder_point_cloud = left_shoulder_point_cloud,
            right_shoulder_rgb = right_shoulder_rgb,
            right_shoulder_depth = right_shoulder_depth,
            right_shoulder_mask = right_shoulder_mask,
            right_shoulder_point_cloud = right_shoulder_point_cloud,
            overhead_rgb = overhead_rgb,
            overhead_depth = overhead_depth,
            overhead_mask = overhead_mask,
            overhead_point_cloud = overhead_point_cloud,
            wrist_rgb = wrist_rgb,
            wrist_depth = wrist_depth,
            wrist_mask = wrist_mask,
            wrist_point_cloud = wrist_point_cloud,
            front_rgb = front_rgb,
            front_depth = front_depth,
            front_mask = front_mask,
            front_point_cloud = front_point_cloud,
            joint_forces = joint_forces,
            
            gripper_matrix = gripper_matrix,
            gripper_joint_positions = gripper_joint_positions,
            gripper_touch_forces = gripper_touch_forces,
            task_low_dim_state = task_low_dim_state,
            ignore_collisions = ignore_collisions,
            misc = misc,
        )
        