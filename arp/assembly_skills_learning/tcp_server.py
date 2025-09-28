import socket
import threading
import queue
import time
import torch
import numpy as np
from utils import configurable, DictConfig, config_to_dict
# from autoregressive_policy import Policy
import clip
import warnings
from dataset import SKILL_TO_ID
from utils.data_util import *
import numpy as np
from torch import nn
from utils.structure import ActResult
from utils.connection import Connection, ConnectionQueues
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from utils.logging_config import get_logger, log_prefix
logger = get_logger(__name__, False)

class NullModel(nn.Module):
    def __init__(self):
        pass
    def forward(self, obj):
        return 
    def act_tcp(self, obj):
        return ActResult(action=np.random.random((0,7)))
    
class ARP_TCP_Client:
    def __init__(self, cfg:DictConfig, debug_mode:bool=False, visualize_pc:bool=False):
        self.debug_mode = debug_mode
        self.host = cfg.tcp.host
        self.port = cfg.tcp.port
        self.max_queue = cfg.tcp.max_queue_size
        self.max_retries = cfg.tcp.max_retries
        self.retry_interval = cfg.tcp.retry_interval
        self.max_client_num = cfg.tcp.max_client_num
        self.queue_time_out = cfg.tcp.queue_time_out
        self.cfg = cfg
        self.time_in_state =cfg.env.time_in_state if hasattr(cfg.env, 'time_in_state') else False
        self.episode_length =cfg.env.episode_length if hasattr(cfg.env, 'episode_length') else 8
        self.device = cfg.eval.device
        self.visualize_pc:bool = visualize_pc
        
        """Load models"""
        if self.debug_mode:
            self.clip_model = NullModel()
            self.model = NullModel()
        else:
            # Load the CLIP model (ViT-B/32 is common)
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            # self.model:Policy = self.setup_model(cfg)
            self.model = self.setup_model(cfg)
        
        """Socket and Connection set up"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow the socket to reuse the address (even if occupied)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.1)# second
        self.connections = ConnectionQueues(queue_time_out=self.queue_time_out)


        """Define stop event and threadings"""
        self.stop_event = threading.Event()
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.model_thread = threading.Thread(target=self._model_loop, daemon=True)
        self.accept_thread = threading.Thread(target=self._listen_loop, daemon=True)
        
        
        """Try to connect to client"""
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        logger.info(f"{log_prefix('Setup')} Listening on {self.host}:{self.port}...")
        
        """Start threads"""
        self.send_thread.start()
        self.process_thread.start()
        self.model_thread.start()
        self.accept_thread.start()
        
            
    def __del__(self):
        logger.info('Release socket.')
        if hasattr(self, 'sock') and self.sock:
            self.stop_event.set()
            self.send_thread.join(timeout=1)
            self.process_thread.join(timeout=1)
            self.model_thread.join(timeout=1)
            self.accept_thread.join(timeout=1)
        for conn in self.connections:
            conn:Connection
            conn.close()
    
    def setup_model(self, cfg:DictConfig):
        logger.info(f'{log_prefix("Setup")} Start set up model')
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()

        # Set up the model
        py_module = cfg.py_module
        from importlib import import_module
        MOD = import_module(py_module)
        Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork
        net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{self.device}").to(self.device)
        agent = Policy(net, cfg.model.hp)
        agent.build(training=False, device=self.device)

        # Load weights
        logger.info(f'{log_prefix("Setup")} Loading Model weights from {cfg.model.weights}')
        agent.load(cfg.model.weights)    
        logger.info(f'{log_prefix("Setup")} Loaded Model')
        
        # Set up evaluation
        agent.eval()
        torch.set_grad_enabled(False)
        logger.info(f'{log_prefix("Setup")} Finish setting up model')
        
        return agent


    def run_forever(self):
        try:
            while not self.stop_event.is_set():                
                time.sleep(5)
                # for conn in self.connections:
                #     logger.info(f'{log_prefix("Main", conn.print_address)} | send_queue:{conn.send_queue.qsize()} | recv_queue:{conn.recv_queue.qsize()} | proc_queue:{conn.proc_queue.qsize()}')
        except KeyboardInterrupt:
            logger.info(f"{log_prefix('Main')} Interrupted. Shutting down...")
        except Exception as e:
            logger.error(f'{log_prefix("Main")} Error: {e}')
            self.stop_event.set()

    def _listen_loop(self):
        while not self.stop_event.is_set():
            try:                
                conn, addr = self.sock.accept()
                connection = Connection(conn=conn, max_queue=self.max_queue, address=addr)
                self.connections.put(connection)
                logger.info(f"{log_prefix('Listen')} Accepted connection from {addr}")
            except socket.timeout:
                continue  # Expected timeout for stop_event checks
            except Exception as e:
                logger.error(f"{log_prefix('Listen')} Error during setup: {e}")
                self.stop_event.set()

    def _send_loop(self):
        while not self.stop_event.is_set():
            for connection in self.connections:
                try:
                    connection:Connection
                    connection.send()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"{log_prefix('Send', connection.addr)} | Error: {e}")
                    connection.close()



    def _process_loop(self):
        logger.debug('Start Process Loop')
        while not self.stop_event.is_set():
            for connection in self.connections:
                try:
                    connection:Connection
                    msg = connection.recv_queue.get(timeout=self.queue_time_out)
                    
                    logger.info(f"{log_prefix('Process', connection.addr)} | Start  frame {msg['frame_idx']}")
                    msg['gripper_pose'] = np.array(msg['gripper_pose']).squeeze()
                    # 提前转为正确格式
                    msg['gripper_pose'] = axis_angle_to_quaternion_pose(msg['gripper_pose'][:-1])
                    msg['gripper_open'] =  float(msg['gripper_pose'][-1] > 0.5) # TODO 这里原先写< 0.5 打个断点确认一下

                    obs = {}
                    obs['task'] = msg['task']
                    logger.info('task: {}'.format(msg['task']))
                    obs['task_idx'] = SKILL_TO_ID[msg['task']]
                    obs['frame_idx'] = msg['frame_idx']
                    obs['description'] = msg['description']
                    # Get Optional Parameters
                    obs['variation_idx'] = msg.get('variation_idx')
                    obs['robot'] = msg.get('robot')
                    # Process message and save results to obs 

                    obs['gripper_pose'] = torch.tensor(msg['gripper_pose'], dtype=torch.float32).to(self.device)
                    obs['gripper_open'] = torch.tensor(msg['gripper_open'], dtype=torch.float32).to(self.device)
                    # 这里传的是msg，所以msg里的pose要提前转换为四元数，open也要转
                    # TODO 如果要做时间步编码，需要维护一个当前kp,这里使用obs['frame_idx']

                    obs['low_dim_state'] = get_low_dim_state(self.cfg.env.origin_style_state, msg, include_time_in_state=self.time_in_state,episode_length=self.episode_length).view(1, -1)
                    obs['lang_goal_embs'] = get_lang_emb(msg, self.cfg.model.hp.add_lang, self.debug_mode, self.clip_model, self.device)
                    # Get obs rgb, depth, pointclouds
                    append_rgbd_pc_to_obs(msg, obs, self.visualize_pc)
                    connection.proc_queue.put(obs)
                    logger.info(f"{log_prefix('Process', connection.addr)} | Finish frame {msg['frame_idx']}")
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"{log_prefix('Process', connection.addr)} | Error: {e}")
                    traceback.print_exc()
                    connection.close()

    def _model_loop(self):
        """Simulate a model that generates messages to send."""
        while not self.stop_event.is_set():
            for connection in self.connections:
                try:
                    connection:Connection
                    msg = connection.proc_queue.get(timeout=self.queue_time_out)
                    start_time = time.time() 
                    logger.info(f"{log_prefix('Model', connection.addr)} | Start  frame {msg['frame_idx']}")

                    for k, v in msg.items():
                        if isinstance(v, torch.Tensor):
                            msg[k] = v.to(self.device)
                    result = self.model.act_tcp(msg)
                    send_msg = self._build_send_message(msg, result)
                    
                    connection.send_queue.put(send_msg)

                    duration = time.time() - start_time
                    logger.info(f"{log_prefix('Model', connection.addr)} | finish frame {msg['frame_idx']} | Inference time: {duration:3f}s | \nSend pose: {send_msg['gripper_pose']}")
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"{log_prefix('Model', connection.addr)} | Error: {e}")
                    traceback.print_exc()
                    connection.close()

    ##region model utils
    def _build_send_message(self, msg, result):
        if self.debug_mode: return { "task": msg['task'], "frame_idx": msg['frame_idx'], "gripper_pose": [0.0]*7 }
        
        pose = quaternion_pose_to_axis_angle(result.action[:7])
        pose_7d = np.concatenate([pose, [result.action[-2]]], 0).tolist()
        diffs = [result.action[:7].tolist()[i] - msg['gripper_pose'].cpu().numpy().tolist()[i] for i in range(7)]
        logger.info(f"{log_prefix('Model')} | gripper_pose diff {diffs}")

        return {
            "task": msg['task'],  
            "frame_idx": msg['frame_idx'],  
            "gripper_pose": pose_7d,  
        } 
    ##endregion

# @configurable('./configs/act.yaml')
# @configurable('./configs/rvt2.yaml')
@configurable('./configs/arp_plus_tcp.yaml')
def main(cfg:DictConfig):
    client = ARP_TCP_Client(cfg, debug_mode=False, visualize_pc=False)
    client.run_forever()

if __name__ == "__main__":
    main()
