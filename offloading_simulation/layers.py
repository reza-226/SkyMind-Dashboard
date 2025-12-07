"""
تعریف لایه‌های محاسباتی برای Offloading
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ComputationalLayer:
    """کلاس پایه برای لایه‌های محاسباتی"""
    name: str
    processing_power: float  # GFLOPS
    energy_per_operation: float  # Joules per GFLOPS
    base_latency: float  # ms
    bandwidth: float  # Mbps
    max_capacity: int  # تعداد Task همزمان
    
    def calculate_processing_time(self, task_complexity: float) -> float:
        """محاسبه زمان پردازش (ms)"""
        return (task_complexity / self.processing_power) * 1000
    
    def calculate_energy(self, task_complexity: float) -> float:
        """محاسبه انرژی مصرفی (Joules)"""
        return task_complexity * self.energy_per_operation
    
    def calculate_transmission_time(self, data_size_mb: float) -> float:
        """محاسبه زمان انتقال داده (ms)"""
        return (data_size_mb * 8 / self.bandwidth) * 1000
    
    def calculate_total_latency(self, task_complexity: float, data_size_mb: float) -> float:
        """محاسبه تاخیر کل"""
        proc_time = self.calculate_processing_time(task_complexity)
        trans_time = self.calculate_transmission_time(data_size_mb)
        return self.base_latency + proc_time + trans_time


class LayerFactory:
    """کارخانه سازنده لایه‌ها"""
    
    @staticmethod
    def create_ground_layer() -> ComputationalLayer:
        """لایه Ground (دستگاه محلی)"""
        return ComputationalLayer(
            name="Ground",
            processing_power=2.0,  # 2 GFLOPS
            energy_per_operation=0.5,  # بیشترین مصرف
            base_latency=0.0,  # بدون تاخیر شبکه
            bandwidth=float('inf'),  # محلی
            max_capacity=5
        )
    
    @staticmethod
    def create_edge_layer() -> ComputationalLayer:
        """لایه Edge (سرور لبه)"""
        return ComputationalLayer(
            name="Edge",
            processing_power=10.0,  # 10 GFLOPS
            energy_per_operation=0.2,
            base_latency=5.0,  # 5ms تاخیر شبکه
            bandwidth=100.0,  # 100 Mbps
            max_capacity=20
        )
    
    @staticmethod
    def create_fog_layer() -> ComputationalLayer:
        """لایه Fog (محاسبات مه)"""
        return ComputationalLayer(
            name="Fog",
            processing_power=50.0,  # 50 GFLOPS
            energy_per_operation=0.1,
            base_latency=20.0,  # 20ms
            bandwidth=500.0,  # 500 Mbps
            max_capacity=100
        )
    
    @staticmethod
    def create_cloud_layer() -> ComputationalLayer:
        """لایه Cloud (ابر)"""
        return ComputationalLayer(
            name="Cloud",
            processing_power=200.0,  # 200 GFLOPS
            energy_per_operation=0.05,  # کمترین مصرف
            base_latency=50.0,  # 50ms
            bandwidth=1000.0,  # 1 Gbps
            max_capacity=1000
        )
    
    @staticmethod
    def create_all_layers() -> Dict[str, ComputationalLayer]:
        """ایجاد همه لایه‌ها"""
        return {
            "ground": LayerFactory.create_ground_layer(),
            "edge": LayerFactory.create_edge_layer(),
            "fog": LayerFactory.create_fog_layer(),
            "cloud": LayerFactory.create_cloud_layer()
        }
