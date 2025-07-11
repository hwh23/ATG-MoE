from multiprocessing import Value
import numpy as np
import torch
from utils.structure import Agent, Env, FullTransition, ActResult
from utils_with_rlbench import TASK_TO_ID

class RolloutGenerator(object):

    def __init__(self, device = 'cuda:0', use_custom_obs=False):
        self._env_device = device
        self.use_custom_obs = use_custom_obs

    def _get_type(self, x):
        if not hasattr(x, 'dtype'): return np.float32
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, 
                env: Env, agent: Agent,
                episode_length: int,
                eval: bool, eval_seed: int = 0,
                record_enabled: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_seed)
        else:
            obs = env.reset()
        task={v:k for k,v in env._task_name_to_idx.items()}[env._active_task_id]
        agent.reset(task=task, desc=env._lang_goal)

        obs_history = {k: np.array(v, dtype=self._get_type(v)) if not isinstance(v, dict) else v for k, v in obs.items()}
        for step in range(episode_length):
            # add batch dimension
            prepped_data = {k: torch.tensor(v, device=self._env_device)[None, ...] if not isinstance(v, dict) else v 
                            for k, v in obs_history.items()}
            int_task_id = TASK_TO_ID[task]
            prepped_data["task_idx"] = int_task_id
            # import pudb;
            # pudb.set_trace()
            act_result = agent.act(step_signal.value, prepped_data)

            if act_result is None:
                return

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = self.get_obs(transition)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k] = self.get_obs(transition)[k]

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = FullTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) if not isinstance(v, dict) else v
                                    for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene, steps=60, step_scene=True)
            obs = self.get_obs(transition)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
            
    def get_obs(self, transition: FullTransition):
        if self.use_custom_obs:
            # TODO : Handle custom observation logic
            # This is a placeholder for custom observation handling.
            # obs.keys()
            # dict_keys(['left_shoulder_rgb', 'left_shoulder_mask', 'left_shoulder_point_cloud', # shape 3, 128, 128
            # 'right_shoulder_rgb', 'right_shoulder_mask', 'right_shoulder_point_cloud',  # shape 3, 128, 128
            # 'wrist_rgb', 'wrist_mask', 'wrist_point_cloud',  # shape 3, 128, 128
            # 'front_rgb', 'front_mask', 'front_point_cloud',  # shape 3, 128, 128
            # 'ignore_collisions',  # shape 1, (1. or 0.)
            # 'low_dim_state', # shape (3,)
            # 'left_shoulder_camera_extrinsics', # (4, 4)
            # 'left_shoulder_camera_intrinsics', # (3, 3)
            # 'right_shoulder_camera_extrinsics', 'right_shoulder_camera_intrinsics', 
            # 'front_camera_extrinsics', 'front_camera_intrinsics', 'wrist_camera_extrinsics', 
            # 'wrist_camera_intrinsics', 
            # 'lang_goal_tokens', # shape (77,)
            # 'gripper_pose', # shape (7,)
            # 'gripper_open' # float, 0. or 1.
            # ])
            obs = None
        else:
            obs = dict(transition.observation)
        return obs