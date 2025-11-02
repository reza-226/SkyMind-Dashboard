# test_core_modules.py
from core.network import ChannelModel
from core.taskgen import TaskGenerator
from core.queue import MMcQueue
import numpy as np

print("\n=== Test 1: Air-to-Ground Channel & SNR ===")
channel = ChannelModel(frequency_ghz=2.4, shadowing_db=2.0)
distance = 1000  # meters
tx_power = 20    # dBm
snr = channel.snr(tx_power, distance)
print(f"SNR at {distance} m = {snr:.2f} dB")

print("\n=== Test 2: Task DAG Generation ===")
tg = TaskGenerator(num_tasks=5)
dag = tg.generate_task_dag()
profiles = tg.generate_task_profiles()
print("Generated DAG:", dag)
for i, p in enumerate(profiles):
    print(f"Task {i}: Size={p['size']:.2f} MB, Cycles={p['cycles']:.1f} MHz")

print("\n=== Test 3: M/M/c Queue Performance ===")
queue = MMcQueue(arrival_rate=4, service_rate=2, servers=3)
print(f"Utilization ρ = {queue.utilization():.3f}")
print(f"Erlang-C Probability = {queue.erlang_c():.3f}")
print(f"E[Waiting] = {queue.expected_waiting_time():.3f} s")
print(f"E[Response] = {queue.expected_response_time():.3f} s")
