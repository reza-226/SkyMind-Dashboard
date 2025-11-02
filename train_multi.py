# ============================================================
# File: train_multi.py
# SkyMind‑TPSG Training Framework  |  Integrated Scientific Version
# ============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

N_AGENTS, N_USERS = 3, 3
env = MultiUAVEnv(n_agents=N_AGENTS, n_users=N_USERS)

# derive correct state dimension dynamically
dummy_state = env.reset()
def flatten_state(sd):
    return np.concatenate([
        sd["uav_positions"].flatten(),
        sd["energy"].flatten(),
        sd["distances"].flatten()
    ])
STATE_DIM = len(flatten_state(dummy_state))   # ✅ automatic dimension (e.g., 18)
ACTION_DIM = 4

agents = [MADDPG_Agent(state_dim=STATE_DIM,
                       action_dim=ACTION_DIM,
                       n_agents=N_AGENTS)
          for _ in range(N_AGENTS)]

EPISODES, MAX_STEPS = 300, 50
episode_rewards, average_delays, total_energy = [], [], []

print(f"[SkyMind‑TPSG] Derived STATE_DIM={STATE_DIM}")
print("\n[SkyMind‑TPSG] Training Multi‑Agent DRL Simulation started...\n")

for ep in range(EPISODES):
    state = env.reset()
    ep_reward, ep_delays, ep_energy = 0.0, [], []

    for step in range(MAX_STEPS):
        actions = []
        for ag in agents:
            s_i = flatten_state(state)
            a_i = ag.act(s_i, noise_scale=0.1)
            actions.append(a_i)

        next_state, rewards, done, info = env.step_multi(actions)
        for ag in agents:
            ag.update(None, agents, batch_size=128)

        ep_reward += np.mean(rewards)
        ep_delays.append(np.mean(info["delays"]))
        ep_energy.append(np.mean(info["energy_total"]))
        state = next_state

        if done:
            break

    avg_delay = np.mean(ep_delays)
    avg_energy = np.sum(ep_energy)
    episode_rewards.append(ep_reward)
    average_delays.append(avg_delay)
    total_energy.append(avg_energy)

    print(f"Episode {ep+1:03d} | Reward: {ep_reward:8.3f} "
          f"| AvgDelay(ms): {avg_delay:7.2f} | Energy(J): {avg_energy:10.2f}")

# save models and curves
for i, ag in enumerate(agents):
    torch.save(ag.actor.state_dict(), f"models/actor_agent{i}.pt")
np.savez("results/training_metrics.npz",
         episode_rewards=episode_rewards,
         average_delays=average_delays,
         total_energy=total_energy)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(episode_rewards)
plt.subplot(3, 1, 2)
plt.plot(average_delays)
plt.subplot(3, 1, 3)
plt.plot(total_energy)
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=300)
plt.close()
print("\n--- End of Simulation ---")
