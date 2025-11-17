"""
Multi-UAV Environment with Dictionary Observation/Action Spaces
Compatible with MADDPG and Gymnasium Multi-Agent API
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import logging

# Import custom modules
from core.obstacles import Obstacle, ObstacleManager
from core.collision_checker import CollisionChecker

logger = logging.getLogger(__name__)


class MultiUAVEnv(gym.Env):
    """
    Multi-Agent UAV Environment with DAG-aware task offloading
    
    Features:
    - Dictionary observation and action spaces per agent
    - Obstacle management and collision detection
    - Task queue system with energy constraints
    - Supports both local processing and edge offloading
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(
        self,
        num_uavs: int = 3,
        num_ground_users: int = 5,
        map_size: Tuple[float, float, float] = (1000.0, 1000.0, 200.0),
        max_steps: int = 500,
        num_obstacles: int = 10,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize Multi-UAV Environment
        
        Args:
            num_uavs: Number of UAV agents
            num_ground_users: Number of ground users generating tasks
            map_size: (x_max, y_max, z_max) boundaries
            max_steps: Maximum steps per episode
            num_obstacles: Number of obstacles in environment
            seed: Random seed for reproducibility
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()
        
        # Environment parameters
        self.num_uavs = num_uavs
        self.num_ground_users = num_ground_users
        
        # ✅ اصلاح: پشتیبانی از ورودی 2D و 3D
        if isinstance(map_size, (list, tuple, np.ndarray)):
            map_size = np.array(map_size, dtype=np.float32)
            if len(map_size) == 2:
                # اگر فقط [X, Y] داده شد، Z را به صورت پیش‌فرض اضافه می‌کنیم
                self.map_size = np.array([map_size[0], map_size[1], 200.0], dtype=np.float32)
                logger.info(f"Map size extended from 2D to 3D: {self.map_size}")
            elif len(map_size) == 3:
                self.map_size = map_size
            else:
                raise ValueError(f"map_size must be 2D or 3D, got shape {map_size.shape}")
        else:
            raise TypeError(f"map_size must be array-like, got {type(map_size)}")
        
        # استخراج ابعاد برای سازگاری
        self.map_size_2d = self.map_size[:2]  # برای موانع 2D
        self.map_size_3d = self.map_size  # برای محدودیت‌های پرواز 3D
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Set random seed
        if seed is not None:
            self.seed(seed)
        
        # ✅ اصلاح: ارسال map_size_2d به ObstacleManager
        self.obstacle_manager = ObstacleManager(
            map_size=self.map_size_2d,  # فقط X, Y
            num_obstacles=num_obstacles
        )
        self.collision_checker = CollisionChecker(
            obstacle_manager=self.obstacle_manager
        )
        
        # UAV parameters
        self.max_velocity = 15.0  # m/s
        self.max_acceleration = 5.0  # m/s^2
        self.uav_radius = 2.0  # meters
        self.min_altitude = 20.0
        
        # ✅ اصلاح: استفاده ایمن از ارتفاع
        self.max_altitude = float(self.map_size_3d[2])
        
        # Energy parameters
        self.max_battery = 100.0  # Wh
        self.hover_power = 10.0  # W
        self.move_power = 15.0  # W
        self.compute_power = 20.0  # W per unit computation
        
        # Communication parameters
        self.bandwidth = 10e6  # 10 MHz
        self.noise_power = 1e-13  # W
        self.path_loss_exponent = 2.5
        self.reference_distance = 1.0  # meters
        
        # Task parameters
        self.task_arrival_rate = 0.3  # tasks per step
        self.max_task_size = 1000.0  # KB
        self.max_cpu_cycles = 1e9  # cycles
        self.max_queue_size = 10
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Initialize state variables
        self.current_step = 0
        self.uav_positions = None
        self.uav_velocities = None
        self.uav_batteries = None
        self.ground_user_positions = None
        self.task_queues = None
        
    def seed(self, seed: int = None) -> List[int]:
        """Set random seed for reproducibility"""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def _define_spaces(self):
        """Define observation and action spaces as dictionaries"""
        
        # Single agent observation space
        # [x, y, z, vx, vy, vz, battery, local_tasks, nearby_uavs_info, user_distances]
        obs_dim = (
            3 +  # position (x, y, z)
            3 +  # velocity (vx, vy, vz)
            1 +  # battery level
            1 +  # number of tasks in queue
            (self.num_uavs - 1) * 4 +  # other UAVs: (relative_x, relative_y, relative_z, distance)
            self.num_ground_users * 2  # users: (distance, has_task)
        )
        
        single_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Single agent action space
        # [acc_x, acc_y, acc_z, offload_decision, resource_allocation]
        single_action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Create dictionary spaces for all agents
        self.observation_space = spaces.Dict({
            f'agent_{i}': single_obs_space for i in range(self.num_uavs)
        })
        
        self.action_space = spaces.Dict({
            f'agent_{i}': single_action_space for i in range(self.num_uavs)
        })
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Returns:
            observations: Dict of observations for each agent
            info: Dict of additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        
        # Initialize UAV positions (random but safe)
        self.uav_positions = np.zeros((self.num_uavs, 3), dtype=np.float32)
        for i in range(self.num_uavs):
            valid_position = False
            attempts = 0
            while not valid_position and attempts < 100:
                pos = np.array([
                    self.np_random.uniform(0, self.map_size_3d[0]),
                    self.np_random.uniform(0, self.map_size_3d[1]),
                    self.np_random.uniform(self.min_altitude, self.max_altitude)
                ], dtype=np.float32)
                
                if not self.collision_checker.check_collision(pos, self.uav_radius):
                    self.uav_positions[i] = pos
                    valid_position = True
                attempts += 1
            
            if not valid_position:
                logger.warning(f"Could not find collision-free position for UAV {i}")
                self.uav_positions[i] = pos  # Use last attempt
        
        # Initialize velocities and batteries
        self.uav_velocities = np.zeros((self.num_uavs, 3), dtype=np.float32)
        self.uav_batteries = np.full(self.num_uavs, self.max_battery, dtype=np.float32)
        
        # Initialize ground user positions
        self.ground_user_positions = np.zeros((self.num_ground_users, 2), dtype=np.float32)
        for i in range(self.num_ground_users):
            self.ground_user_positions[i] = [
                self.np_random.uniform(0, self.map_size_2d[0]),
                self.np_random.uniform(0, self.map_size_2d[1])
            ]
        
        # Initialize task queues
        self.task_queues = [[] for _ in range(self.num_uavs)]
        
        # Reset obstacle manager
        self.obstacle_manager.reset()
        
        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents"""
        observations = {}
        
        for i in range(self.num_uavs):
            obs_parts = []
            
            # Own state: position, velocity, battery
            obs_parts.append(self.uav_positions[i])
            obs_parts.append(self.uav_velocities[i])
            obs_parts.append(np.array([self.uav_batteries[i]], dtype=np.float32))
            obs_parts.append(np.array([len(self.task_queues[i])], dtype=np.float32))
            
            # Other UAVs information
            for j in range(self.num_uavs):
                if i != j:
                    relative_pos = self.uav_positions[j] - self.uav_positions[i]
                    distance = np.linalg.norm(relative_pos)
                    obs_parts.append(np.concatenate([relative_pos, [distance]]))
            
            # Ground users information
            for j in range(self.num_ground_users):
                user_pos_3d = np.array([
                    self.ground_user_positions[j, 0],
                    self.ground_user_positions[j, 1],
                    0.0
                ], dtype=np.float32)
                distance = np.linalg.norm(self.uav_positions[i] - user_pos_3d)
                has_task = 1.0 if self.np_random.rand() < self.task_arrival_rate else 0.0
                obs_parts.append(np.array([distance, has_task], dtype=np.float32))
            
            # Concatenate all parts
            observations[f'agent_{i}'] = np.concatenate(obs_parts).astype(np.float32)
        
        return observations
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions: Dictionary of actions for each agent
                     Each action: [acc_x, acc_y, acc_z, offload_decision, resource_allocation]
        
        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent
            terminated: Dict of termination flags for each agent
            truncated: Dict of truncation flags for each agent
            info: Dict of additional information
        """
        self.current_step += 1
        
        rewards = {}
        terminated = {}
        truncated = {}
        
        # Process actions for each UAV
        for i in range(self.num_uavs):
            agent_key = f'agent_{i}'
            action = actions[agent_key]
            
            # Extract action components
            acceleration = action[:3] * self.max_acceleration
            offload_decision = action[3]  # 0: local, 1: offload
            resource_allocation = action[4]  # proportion of resources
            
            # Update velocity and position
            self.uav_velocities[i] += acceleration * 0.1  # dt = 0.1s
            self.uav_velocities[i] = np.clip(
                self.uav_velocities[i],
                -self.max_velocity,
                self.max_velocity
            )
            
            new_position = self.uav_positions[i] + self.uav_velocities[i] * 0.1
            
            # ✅ اصلاح: استفاده از map_size_3d برای محدودیت‌های مرزی
            new_position = np.clip(
                new_position,
                [0, 0, self.min_altitude],
                [self.map_size_3d[0], self.map_size_3d[1], self.max_altitude]
            )
            
            # Check collision
            collision = self.collision_checker.check_collision(new_position, self.uav_radius)
            
            if not collision:
                self.uav_positions[i] = new_position
            
            # Energy consumption
            movement_energy = self.move_power * np.linalg.norm(self.uav_velocities[i]) * 0.1
            hover_energy = self.hover_power * 0.1
            
            # Process tasks
            task_energy = 0.0
            task_delay = 0.0
            tasks_processed = 0
            
            if len(self.task_queues[i]) > 0:
                task = self.task_queues[i][0]
                
                if offload_decision < 0.5:  # Local processing
                    processing_time = task['cpu_cycles'] / (1e9 * resource_allocation + 1e-6)
                    task_energy = self.compute_power * processing_time
                    task_delay = processing_time
                else:  # Offload to edge
                    # Find nearest ground user
                    distances = np.linalg.norm(
                        self.ground_user_positions - self.uav_positions[i, :2],
                        axis=1
                    )
                    nearest_user = np.argmin(distances)
                    distance = distances[nearest_user]
                    
                    # Calculate transmission time and energy
                    path_loss = self.reference_distance / (distance + 1e-6) ** self.path_loss_exponent
                    snr = path_loss / self.noise_power
                    data_rate = self.bandwidth * np.log2(1 + snr)
                    transmission_time = task['data_size'] * 1024 * 8 / (data_rate + 1e-6)
                    
                    # Edge processing time (assumed faster)
                    edge_processing_time = task['cpu_cycles'] / (5e9 * resource_allocation + 1e-6)
                    
                    task_energy = 0.5 * transmission_time  # Simplified transmission energy
                    task_delay = transmission_time + edge_processing_time
                
                # Update battery
                total_energy = movement_energy + hover_energy + task_energy
                self.uav_batteries[i] -= total_energy / 3600.0  # Convert to Wh
                
                # Mark task as processed
                if self.uav_batteries[i] > 0:
                    self.task_queues[i].pop(0)
                    tasks_processed = 1
            else:
                total_energy = movement_energy + hover_energy
                self.uav_batteries[i] -= total_energy / 3600.0
            
            # Calculate reward
            reward = 0.0
            
            # Reward for processing tasks
            reward += tasks_processed * 10.0
            
            # Penalty for delay
            reward -= task_delay * 0.1
            
            # Penalty for energy consumption
            reward -= total_energy * 0.01
            
            # Penalty for collision
            if collision:
                reward -= 50.0
            
            # Penalty for low battery
            if self.uav_batteries[i] < 10.0:
                reward -= 20.0
            
            rewards[agent_key] = reward
            
            # Check termination conditions
            terminated[agent_key] = (
                self.uav_batteries[i] <= 0 or
                collision
            )
        
        # Global truncation (episode length)
        is_truncated = self.current_step >= self.max_steps
        truncated = {f'agent_{i}': is_truncated for i in range(self.num_uavs)}
        
        # Get new observations
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        return {
            'step': self.current_step,
            'uav_batteries': self.uav_batteries.copy(),
            'task_queue_lengths': [len(q) for q in self.task_queues],
            'uav_positions': self.uav_positions.copy()
        }
    
    def render(self):
        """Render environment (placeholder)"""
        if self.render_mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"UAV Positions:\n{self.uav_positions}")
            print(f"UAV Batteries: {self.uav_batteries}")
            print(f"Task Queues: {[len(q) for q in self.task_queues]}")
    
    def close(self):
        """Clean up resources"""
        pass


# Example usage
if __name__ == "__main__":
    # ✅ تست با ورودی 2D
    print("=" * 60)
    print("Testing with 2D map_size input:")
    print("=" * 60)
    env_2d = MultiUAVEnv(
        num_uavs=3,
        num_ground_users=5,
        map_size=[1000.0, 1000.0],  # فقط X, Y
        seed=42
    )
    print(f"Map size (internal 3D): {env_2d.map_size_3d}")
    print(f"Map size (2D for obstacles): {env_2d.map_size_2d}")
    print(f"Max altitude: {env_2d.max_altitude}")
    
    obs_2d, info_2d = env_2d.reset()
    print("✅ 2D input test passed!\n")
    
    # ✅ تست با ورودی 3D
    print("=" * 60)
    print("Testing with 3D map_size input:")
    print("=" * 60)
    env_3d = MultiUAVEnv(
        num_uavs=3,
        num_ground_users=5,
        map_size=(1000.0, 1000.0, 200.0),  # X, Y, Z
        seed=42
    )
    print(f"Map size (internal 3D): {env_3d.map_size_3d}")
    print(f"Map size (2D for obstacles): {env_3d.map_size_2d}")
    print(f"Max altitude: {env_3d.max_altitude}")
    
    obs_3d, info_3d = env_3d.reset()
    print("✅ 3D input test passed!\n")
    
    # تست یک قدم
    print("=" * 60)
    print("Testing step execution:")
    print("=" * 60)
    actions = {
        agent_id: env_3d.action_space[agent_id].sample()
        for agent_id in env_3d.action_space.keys()
    }
    
    obs, rewards, terminated, truncated, info = env_3d.step(actions)
    
    print("Rewards:", rewards)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("✅ Step execution test passed!")
