import json
import pickle
import numpy as np
import cv2
import zlib
import struct
from .logging_config import get_logger
logger = get_logger(__name__, False)


def send_pickle(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(data)
    length = struct.pack('>I', len(compressed))
    conn.sendall(length + compressed)
    
def send_json(conn, obj):
    data = json.dumps(obj)
    msg_bytes = data.encode('utf-8')
    length = struct.pack('<I', len(msg_bytes))
    conn.sendall(length + msg_bytes)

def recv_pickle(conn):
    raw_len = recv_exact(conn, 4)
    msg_len = struct.unpack('>I', raw_len)[0]
    data = recv_exact(conn, msg_len)
    msg_dict = pickle.loads(zlib.decompress(data))
    return msg_dict

def recv_json(conn):
    raw_len = recv_exact(conn, 4)
    msg_len = struct.unpack('<I', raw_len)[0]        
    data = recv_exact(conn, msg_len)
    msg_dict = json.loads(data.decode('utf-8'))
    images = {}
    for cam_name, cam_info in msg_dict["images"].items():
        rgb_shape = tuple(cam_info["rgb_shape"])      # e.g. (512,512,3)
        depth_shape = tuple(cam_info["depth_shape"])  # e.g. (512,512)
        logger.debug(f"[Recv Json] 图像 {cam_name} 尺寸: RGB {rgb_shape}, Depth {depth_shape}")

        # # ----- 读取 RGB float32 raw 数据 -----
        rgb_len_raw = recv_exact(conn, 4)
        rgb_len = struct.unpack('<I', rgb_len_raw)[0]
        rgb_bytes = recv_exact(conn, rgb_len)
        rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8)
        rgb_img = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        logger.debug(f"[Recv Json] 图像 {cam_name}  RGB解码完毕, 尺寸: {rgb_len}")

        # ----- 读取 Depth float32 raw 数据 -----
        depth_len_raw = recv_exact(conn, 4)
        depth_len = struct.unpack('<I', depth_len_raw)[0]
        depth_bytes = recv_exact(conn, depth_len)
        depth_img = np.frombuffer(depth_bytes, dtype=np.float32)
        if depth_shape[0] != depth_shape[1]:
            logger.error(f"The depth image is not square, reshaping to the non-squared shape {depth_shape}.")
        depth_img = depth_img.reshape(depth_shape)
        logger.debug(f"[Recv Json] 图像 {cam_name} Depth 解码完毕, 尺寸: {depth_len}")

        images[cam_name] = {
            "rgb": rgb_img,
            "depth": depth_img,
        }
    msg_dict["images"] = images
    return msg_dict
    
def recv_exact(conn, length):
    buf = b''
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            raise BrokenPipeError("Socket connection broken during recv()")  # or ConnectionResetError
        buf += chunk
    return buf