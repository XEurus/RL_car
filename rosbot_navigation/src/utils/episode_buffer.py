"""
Episode Buffer - 用于缓存整个episode的经验并进行奖励修改
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import deque

class EpisodeBuffer:
    """
    Episode Buffer类 - 用于缓存整个episode的经验并进行奖励修改
    
    功能：
    1. 缓存整个episode的经验
    2. 在episode结束时，修改每一步的奖励
    3. 添加step-level距离奖励
    4. 添加episode-level距离奖励，平均到每一步
    """
    
    def __init__(self, max_episodes: int = 10):
        """
        初始化Episode Buffer
        
        参数:
            max_episodes: 最大缓存的episode数量
        """
        self.max_episodes = max_episodes
        self.current_episode = []  # 当前正在收集的episode
        self.completed_episodes = deque(maxlen=max_episodes)  # 已完成的episodes
        
    def add_experience(self, obs: np.ndarray, action: np.ndarray, reward: float, 
                      next_obs: np.ndarray, done: bool, info: Dict = None):
        """
        添加单步经验到当前episode
        
        参数:
            obs: 观测
            action: 动作
            reward: 奖励
            next_obs: 下一步观测
            done: 是否结束
            info: 额外信息
        """
        # 确保info是字典
        if info is None:
            info = {}
            
        # 添加经验到当前episode
        self.current_episode.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_obs': next_obs.copy(),
            'done': done,
            'info': info.copy() if info else {}
        })
        
        # 如果episode结束，处理并移动到已完成队列
        if done:
            self._process_episode()
            
    def _process_episode(self):
        """处理完成的episode，修改奖励并移动到已完成队列"""
        if not self.current_episode:
            return
            
        # 计算episode总奖励
        episode_total_reward = sum(exp['reward'] for exp in self.current_episode)
        
        # 计算episode-level距离奖励（基于最后一步与目标的距离）
        last_step = self.current_episode[-1]
        if 'distance_to_target' in last_step['info']:
            final_distance = last_step['info']['distance_to_target']
            # 距离越小，奖励越大
            episode_distance_reward = 300.0 / (1.0 + final_distance)
        else:
            episode_distance_reward = 0.0
            
        # 平均分配到每一步
        per_step_episode_reward = episode_distance_reward / len(self.current_episode)
        
        # 修改每一步的奖励
        for i, exp in enumerate(self.current_episode):
            # 添加episode总奖励的20%
            exp['reward'] += episode_total_reward * 0.2
            
            # 添加平均分配的episode距离奖励
            exp['reward'] += per_step_episode_reward
            
            # 添加step-level距离奖励（如果有距离信息）
            if 'distance_to_target' in exp['info']:
                step_distance = exp['info']['distance_to_target']
                # 距离越小，奖励越大
                step_distance_reward = 100.0 / (1.0 + step_distance)
                exp['reward'] += step_distance_reward
        
        # 将处理后的episode添加到已完成队列
        self.completed_episodes.append(self.current_episode)
        
        # 重置当前episode
        self.current_episode = []
        
    def get_all_experiences(self) -> List[Dict]:
        """
        获取所有已完成episode的经验
        
        返回:
            List[Dict]: 所有经验的列表
        """
        all_experiences = []
        for episode in self.completed_episodes:
            all_experiences.extend(episode)
        return all_experiences
    
    def clear(self):
        """清空buffer"""
        self.current_episode = []
        self.completed_episodes.clear()
        
    def is_empty(self) -> bool:
        """
        检查buffer是否为空
        
        返回:
            bool: buffer是否为空
        """
        return len(self.completed_episodes) == 0 and len(self.current_episode) == 0
    
    def get_episode_count(self) -> int:
        """
        获取已完成的episode数量
        
        返回:
            int: 已完成的episode数量
        """
        return len(self.completed_episodes)
    
    def get_experience_count(self) -> int:
        """
        获取所有经验的数量
        
        返回:
            int: 所有经验的数量
        """
        return sum(len(episode) for episode in self.completed_episodes) + len(self.current_episode)
