class RewardCalculator:
    """Calculate reward based on latency, energy, and success"""
    
    def __init__(self, w_latency=0.4, w_energy=0.3, w_success=0.3):
        self.w_latency = w_latency
        self.w_energy = w_energy
        self.w_success = w_success
    
    def calculate(self, latency, energy, success):
        latency_reward = -latency / 1000.0
        energy_reward = -energy / 10.0
        success_reward = 10.0 if success else -5.0
        
        reward = (
            self.w_latency * latency_reward +
            self.w_energy * energy_reward +
            self.w_success * success_reward
        )
        
        return reward
