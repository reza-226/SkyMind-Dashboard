# test_core_modules.py
from core.network import ChannelModel
from core.taskgen import TaskGenerator
from core.queue import MMcQueue
import numpy as np

# ---------- 1. Test Channel Model ----------
print("=== Test 1: Channel & SNR ===")
ch = ChannelModel(frequency_ghz=2.4, shadowing_db=2.0)
distance = 1000  # meters
tx_power = 20    # dBm
snr_value = ch.snr(tx_power, distance)
print(f"Distance = {distance} m, TX Power = {tx_power} dBm, SNR = {snr_value:.2f} dB\n")

# ---------- 2. Test Task Generation ----------
print("=== Test 2: Task Generation ===")
tg = TaskGenerator(num_tasks=5)
dag = tg.generate_task_dag()
profiles = tg.generate_task_profiles()
print("DAG links ->", dag)
print("Task Profiles ->")
for idx, prof in enumerate(profiles):
    print(f"Task {idx}: size={prof['size']:.2f} MB, cycles={prof['cycles']:.0f} MHz")
print()

# ---------- 3. Test M/M/c Queue Model ----------
print("=== Test 3: M/M/c Queue ===")
arrival_rate = 4     # λ=4 tasks/sec
service_rate = 2     # μ=2 tasks/sec
servers = 3          # c=3
q = MMcQueue(arrival_rate, service_rate, servers)
print(f"Utilization (ρ)           = {q.utilization():.3f}")
print(f"ErlangC Prob (wait>0)    = {q.erlang_c():.3f}")
print(f"Expected Waiting Time (s) = {q.expected_waiting_time():.3f}")
print(f"Expected Response Time(s) = {q.expected_response_time():.3f}")
