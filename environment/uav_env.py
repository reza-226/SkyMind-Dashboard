"""
UAV Offloading Environment
Custom Gymnasium Environment for MADDPG Training
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVEnvironment(gym.Env):
    """
    Custom UAV Environment for Offloading Decision Making
    
    State Space: 537-dim vector
        - Graph embeddings: 256-dim
        - Node embeddings: 256-dim
        - Flat features: 25-dim
    
    Action Space: 11-dim vector
        - Offload decision (one-hot): 5-dim [Ground, Fog, MEC, Cloud, Local]
        - CPU allocation: 1-dim [0, 1]
        - Bandwidth allocation: 3-dim [0, 1] for each layer
        - UAV movement: 2-dim [-1, 1] for (dx, dy)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        num_tasks=100,
        task_complexity='mixed',
        reward_weights=(0.4, 0.3, 0.3),
        max_steps=500,
        render_mode=None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Environment parameters
        self.num_tasks = num_tasks
        self.task_complexity = task_complexity
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reward weights (latency, energy, success)
        self.w_latency, self.w_energy, self.w_success = reward_weights
        
        # Space definitions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(537,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32
        )
        
        # Layer parameters (Ground, Fog, MEC, Cloud, Local)
        self.layer_latencies = {
            'ground': (50, 100),
            'fog': (30, 80),
            'mec': (20, 60),
            'cloud': (100, 300),
            'local': (10, 50)
        }
        
        self.layer_energy = {
            'ground': (5, 15),
            'fog': (3, 10),
            'mec': (2, 8),
            'cloud': (8, 25),
            'local': (1, 5)
        }
        
        self.layer_success_rates = {
            'ground': 0.95,
            'fog': 0.98,
            'mec': 0.99,
            'cloud': 0.97,
            'local': 0.90
        }
        
        # UAV position
        self.uav_position = np.array([0.0, 0.0])
        self.max_distance = 100.0
        
        # Task queue
        self.task_queue = []
        self.current_task = None
        
        # Episode statistics
        self.episode_stats = {
            'total_latency': 0.0,
            'total_energy': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state (Gymnasium style)"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.uav_position = np.array([0.0, 0.0])
        self.task_queue = self._generate_task_queue()
        self.current_task = self.task_queue.pop(0) if self.task_queue else None
        
        self.episode_stats = {
            'total_latency': 0.0,
            'total_energy': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute action and return next state
        
        Args:
            action: np.array of shape (11,)
                [offload_onehot(5), cpu(1), bandwidth(3), movement(2)]
        
        Returns:
            observation: np.array (537,)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        self.current_step += 1
        
        # Decode action
        offload_decision = np.argmax(action[:5])  # 0-4
        cpu_allocation = (action[5] + 1) / 2  # [-1,1] → [0,1]
        bandwidth_allocation = (action[6:9] + 1) / 2  # [-1,1] → [0,1]
        movement = action[9:11]  # [-1,1] for (dx, dy)
        
        # Update UAV position
        self._update_position(movement)
        
        # Execute offloading decision
        latency, energy, success = self._execute_offloading(
            offload_decision,
            cpu_allocation,
            bandwidth_allocation
        )
        
        # Calculate reward
        reward = self._calculate_reward(latency, energy, success)
        
        # Update statistics
        self.episode_stats['total_latency'] += latency
        self.episode_stats['total_energy'] += energy
        if success:
            self.episode_stats['successful_tasks'] += 1
        else:
            self.episode_stats['failed_tasks'] += 1
        
        # Get next task
        if self.task_queue:
            self.current_task = self.task_queue.pop(0)
        else:
            self.current_task = None
        
        # Check if episode is done
        terminated = self.current_task is None
        truncated = self.current_step >= self.max_steps
        
        # Prepare info
        info = {
            'latency': latency,
            'energy': energy,
            'success': success,
            'offload_layer': ['ground', 'fog', 'mec', 'cloud', 'local'][offload_decision],
            'uav_position': self.uav_position.copy(),
            'step': self.current_step,
            'episode_stats': self.episode_stats.copy()
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """Build 537-dim state vector"""
        # Graph embeddings (256-dim)
        graph_features = self._generate_graph_embedding()
        
        # Node embeddings (256-dim)
        node_features = self._generate_node_embedding()
        
        # Flat features (25-dim)
        flat_features = self._generate_flat_features()
        
        state = np.concatenate([graph_features, node_features, flat_features])
        return state.astype(np.float32)
    
    def _generate_graph_embedding(self):
        """Generate graph-level embeddings (256-dim)"""
        if self.current_task is None:
            return np.zeros(256)
        
        # Simulate GNN output based on task properties
        complexity = self.current_task['complexity']
        data_size = self.current_task['data_size']
        
        embedding = np.random.randn(256) * 0.1
        embedding[:10] = complexity / 100.0  # Normalized complexity
        embedding[10:20] = data_size / 1000.0  # Normalized data size
        
        return embedding
    
    def _generate_node_embedding(self):
        """Generate node-level embeddings (256-dim)"""
        if self.current_task is None:
            return np.zeros(256)
        
        # Simulate node features (5 layers × ~50 dims each)
        embedding = np.zeros(256)
        
        for i, layer in enumerate(['ground', 'fog', 'mec', 'cloud', 'local']):
            start_idx = i * 50
            end_idx = start_idx + 50
            
            # Layer availability and load
            embedding[start_idx:start_idx+10] = np.random.rand(10) * 0.5
            # Layer capacity
            embedding[start_idx+10:start_idx+20] = np.random.rand(10) * 0.8
        
        return embedding
    
    def _generate_flat_features(self):
        """Generate flat features (25-dim)"""
        if self.current_task is None:
            return np.zeros(25)
        
        features = np.zeros(25)
        
        # Task features (10-dim)
        features[0] = self.current_task['complexity'] / 100.0
        features[1] = self.current_task['data_size'] / 1000.0
        features[2] = self.current_task['deadline'] / 1000.0
        features[3:8] = np.random.rand(5)  # Task priority, dependencies, etc.
        
        # UAV features (10-dim)
        features[8:10] = self.uav_position / self.max_distance  # Normalized position
        features[10:15] = np.random.rand(5) * 0.5  # Battery, CPU, memory, etc.
        
        # Environment features (5-dim)
        features[15:20] = np.random.rand(5) * 0.3  # Network conditions, weather, etc.
        
        # Remaining tasks
        features[20] = len(self.task_queue) / self.num_tasks
        
        return features
    
    def _generate_task_queue(self):
        """Generate task queue based on complexity"""
        tasks = []
        
        if self.task_complexity == 'simple':
            complexity_range = (10, 30)
            data_range = (50, 200)
        elif self.task_complexity == 'medium':
            complexity_range = (30, 70)
            data_range = (200, 600)
        elif self.task_complexity == 'complex':
            complexity_range = (70, 100)
            data_range = (600, 1000)
        else:  # mixed
            complexity_range = (10, 100)
            data_range = (50, 1000)
        
        for _ in range(self.num_tasks):
            task = {
                'complexity': np.random.uniform(*complexity_range),
                'data_size': np.random.uniform(*data_range),
                'deadline': np.random.uniform(100, 500)
            }
            tasks.append(task)
        
        return tasks
    
    def _update_position(self, movement):
        """Update UAV position"""
        self.uav_position += movement * 5.0  # Scale movement
        self.uav_position = np.clip(
            self.uav_position,
            -self.max_distance,
            self.max_distance
        )
    
    def _execute_offloading(self, layer_idx, cpu_allocation, bandwidth_allocation):
        """
        Simulate offloading execution
        
        Returns:
            latency: float (ms)
            energy: float (J)
            success: bool
        """
        layer_names = ['ground', 'fog', 'mec', 'cloud', 'local']
        layer = layer_names[layer_idx]
        
        # Get layer parameters
        lat_min, lat_max = self.layer_latencies[layer]
        energy_min, energy_max = self.layer_energy[layer]
        success_rate = self.layer_success_rates[layer]
        
        # Calculate latency (affected by CPU and bandwidth)
        base_latency = np.random.uniform(lat_min, lat_max)
        latency = base_latency * (1.0 - 0.3 * cpu_allocation) * (1.0 - 0.2 * np.mean(bandwidth_allocation))
        
        # Calculate energy (affected by CPU)
        base_energy = np.random.uniform(energy_min, energy_max)
        energy = base_energy * (1.0 + 0.5 * cpu_allocation)
        
        # Determine success
        success = np.random.rand() < success_rate
        
        return latency, energy, success
    
    def _calculate_reward(self, latency, energy, success):
        """Calculate reward based on latency, energy, and success"""
        # Normalize latency and energy
        latency_reward = -latency / 1000.0
        energy_reward = -energy / 10.0
        success_reward = 10.0 if success else -5.0
        
        reward = (
            self.w_latency * latency_reward +
            self.w_energy * energy_reward +
            self.w_success * success_reward
        )
        
        return reward
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == "human":
            if self.current_task:
                print(f"Step {self.current_step}: Task complexity={self.current_task['complexity']:.1f}, "
                      f"UAV position={self.uav_position}")
