import socket
import pickle
import zlib
import struct
import time
import numpy as np
from utils import configurable, DictConfig, config_to_dict
from utils.structure import ASSEMBLY_TASKS
from arp.assembly_skills_learning.dataset import TransitionDataset
import json
import cv2
import logging
logger = logging.getLogger(__name__)

def get_loader(cfg:DictConfig):
    env_cfg = cfg.env
    tasks = ASSEMBLY_TASKS
    dataset = TransitionDataset(cfg.train.demo_folder, tasks, cameras=env_cfg.cameras,
            batch_num=1, batch_size=1, scene_bounds=env_cfg.scene_bounds,
            voxel_size=env_cfg.voxel_size, rotation_resolution=env_cfg.rotation_resolution,
            cached_data_path=cfg.train.cached_dataset_path, time_in_state=cfg.env.time_in_state,
            episode_length=cfg.env.episode_length, k2k_sample_ratios=cfg.train.k2k_sample_ratios, 
            origin_style_state=cfg.env.origin_style_state,
            shuffle=True)

    dataloader, _ = dataset.dataloader(num_workers=cfg.train.num_workers, 
                                                pin_memory=False, distributed=False)
    return dataloader

def send_json(sock, obj:dict):
    header_dict, image_dict = obj
    # First send the main message header
    header_data = json.dumps(header_dict).encode('utf-8')
    header_length = struct.pack('<I', len(header_data))
    sock.sendall(header_length + header_data)
    
    # Then send each image's binary data
    for cam_name, img_data in image_dict.items():
        # Send RGB image
        # rgb_success, rgb_encoded = cv2.imencode('.jpg', img_data["rgb"])
        print(img_data["rgb"].shape)
        print(f'{cam_name}, 2')
        rgb_success, rgb_encoded = cv2.imencode('.jpg', img_data["rgb"].squeeze(), 
                    )
        
        if not rgb_success:
            raise ValueError(f"Failed to encode RGB image for {cam_name}")
        
        rgb_bytes = rgb_encoded.tobytes()
        sock.sendall(struct.pack('<I', len(rgb_bytes)) + rgb_bytes)
        
        # Send Depth image (float32)
        depth_bytes = img_data["depth"].astype(np.float32).tobytes()
        sock.sendall(struct.pack('<I', len(depth_bytes)) + depth_bytes)
        


def recv_json(sock):
    """
    Receive a JSON message that was sent using _send_json()
    Matches the protocol: [4-byte length][json data]
    """
    def _recv_exact(sock, length):
        buf = b''
        while len(buf) < length:
            chunk = sock.recv(length - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed prematurely")
            buf += chunk
        return buf
    try:
        # 1. Receive the 4-byte length prefix
        length_bytes = _recv_exact(sock, 4)
        if len(length_bytes) != 4:
            raise ConnectionError("Incomplete length header received")
        
        # 2. Unpack the message length (little-endian unsigned int)
        msg_len = struct.unpack('<I', length_bytes)[0]
        
        # 3. Receive the actual JSON data
        json_data = _recv_exact(sock, msg_len)
        
        # 4. Decode and parse the JSON
        return json.loads(json_data.decode('utf-8'))
        
    except struct.error as e:
        logger.error(f"Failed to unpack message length: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {str(e)}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode UTF-8: {str(e)}")
        raise
    except socket.timeout:
        logger.error("Receive operation timed out")
        raise
    except Exception as e:
        logger.error(f"Unexpected error receiving JSON: {str(e)}")
        raise

def build_dummy_obs(dataloader):
    obs = next(iter(dataloader))
    # obs['left_shoulder_rgb'] /=100
    # obs['overhead_rgb'] /=100
    # obs['front_rgb'] /=100
    image_dict = {
        "left_shoulder": {
            'rgb':obs['left_shoulder_rgb'].squeeze().permute(1,2,0).cpu().numpy(), 
            "depth": obs['left_shoulder_depth'].squeeze().cpu().numpy(),
            },
        "overhead": {
            'rgb':obs['overhead_rgb'].squeeze().permute(1,2,0).cpu().numpy(), 
            "depth": obs['overhead_depth'].squeeze().cpu().numpy(),
            },
        "front": {
            'rgb':obs['front_rgb'].squeeze().permute(1,2,0).cpu().numpy(), 
            "depth": obs['front_depth'].squeeze().cpu().numpy(),
            },
    }
    header = {
        "task": "piston_sleeve_installation",  
        "frame_idx": int(0),  
        "variation_idx": int(0),  # Optional 额外信息
        "robot": "robot_name",  # Optional 额外信息
        "description": "insert the red sealed piston into the sleeve",  
        "gripper_pose": obs['gripper_pose'].cpu().numpy().tolist(),  
        "images": {
            "left_shoulder": {
                "rgb_shape": list(image_dict['left_shoulder']['rgb'].shape),
                "depth_shape": list(image_dict['left_shoulder']['depth'].shape),
                },
            "overhead": {
                "rgb_shape": list(image_dict['overhead']['rgb'].shape),
                "depth_shape": list(image_dict['overhead']['depth'].shape),
                },
            "front": {
                "rgb_shape": list(image_dict['front']['rgb'].shape),
                "depth_shape": list(image_dict['front']['depth'].shape),
                },
            # Add more cameras as needed
        },
        "misc": {
                "left_shoulder": {
                    "intrinsics": [711.1111, 0.0, 256.0, 0.0, 1066.66663, 256.0, 0.0, 0.0, 1.0],  # Placeholder for camera intrinsics
                    "extrinsics": [0.252,0.757,0.087,0.0894012,-0.852069438,0.491942525,0.154847413 ],  # Placeholder for camera extrinsics
                    "near": float(0.1),  # Near clipping plane (meter)
                    "far": float(100.0),  # Far clipping plane (meter)
                    },
                "overhead": {
                    "intrinsics": [711.1111, 0.0, 256.0, 0.0, 1066.66663, 256.0, 0.0, 0.0, 1.0],  # Placeholder for camera intrinsics
                    "extrinsics": [0.007, 0.673, -0.66, 0.4694715, 0.0, 0.0, 0.8829476],  # Placeholder for camera extrinsics
                    "near": float(0.1),  # Near clipping plane (meter)
                    "far": float(100.0),  # Far clipping plane (meter)
                    },
                "front": {
                    "intrinsics": [711.1111, 0.0, 256.0, 0.0, 1066.66663, 256.0, 0.0, 0.0, 1.0],  # Placeholder for camera intrinsics
                    "extrinsics": [0.013, 0.368, -0.975, 0.1564345, 0.0, 0.0, 0.987688363],  # Placeholder for camera extrinsics
                    "near": float(0.1),  # Near clipping plane (meter)
                    "far": float(100.0),  # Far clipping plane (meter)
                    },
            # ...
        },
    } 
    return header, image_dict

def run_client(cfg:DictConfig):
    port = cfg.tcp.port
    host = cfg.tcp.host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"[Client] Listening on {host}:{port}...")
    
    dataloader = get_loader(cfg)
    frame_idx = 0
    try:
        while True:
            obs = build_dummy_obs(dataloader)
            obs[0]['frame_idx'] = frame_idx
            send_json(sock, obs)
            print(f"[Client] Sent frame {frame_idx}")

            # Wait for client model response
            reply = recv_json(sock)
            print(f"[Client] Received frame {reply['frame_idx']}")
            frame_idx += 1
            time.sleep(1)
    except Exception as e:
        print(f"[Client] Error: {e}")
    finally:
        print("[Client] Closing connection.")
        sock.close()

@configurable()
def main(cfg):
    run_client(cfg)
    
@configurable()
def test_dataloader(cfg):
    dataloader = get_loader(cfg)
    for i, obs in enumerate(dataloader):
        print(i)

    
    
if __name__ == '__main__':
    # main()
    test_dataloader()