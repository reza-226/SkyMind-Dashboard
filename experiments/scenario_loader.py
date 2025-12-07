"""
بارگذاری و مدیریت سناریوهای آزمایشی Multi-Tier
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class TierSpec:
    """مشخصات یک Tier محاسباتی"""
    name: str
    processing_capacity: float
    communication_delay: float
    energy_per_flop: float
    reliability: float
    transmission_power: float


@dataclass
class ComplexityLevel:
    """سطح پیچیدگی سناریو"""
    name: str
    num_tasks: int
    num_uavs: int
    task_size_range: tuple  # (min, max) MB
    deadline_range: tuple   # (min, max) seconds


@dataclass
class Scenario:
    """تعریف کامل یک سناریو آزمایشی"""
    id: str
    tier: str
    complexity: str
    tier_specs: TierSpec
    complexity_specs: ComplexityLevel
    description: str


class ScenarioLoader:
    """بارگذاری و مدیریت سناریوها از فایل YAML"""
    
    def __init__(self, config_path: str = "experiments/scenarios_config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        self.tier_specs: Dict[str, TierSpec] = {}
        self.complexity_levels: Dict[str, ComplexityLevel] = {}
        self.scenarios: List[Scenario] = []
        
    def load(self):
        """بارگذاری تنظیمات از فایل YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"فایل پیکربندی یافت نشد: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._load_tiers()
        self._load_complexity_levels()
        self._load_scenarios()
        
        print(f"✅ بارگذاری شد: {len(self.scenarios)} سناریو")
        
    def _load_tiers(self):
        """بارگذاری مشخصات Tierها"""
        tiers_config = self.config['computing_tiers']
        
        for tier_name, tier_data in tiers_config.items():
            specs = tier_data['specs']
            self.tier_specs[tier_name] = TierSpec(
                name=tier_data['name'],
                processing_capacity=specs['processing_capacity'],
                communication_delay=specs['communication_delay'],
                energy_per_flop=specs['energy_per_flop'],
                reliability=specs['reliability'],
                transmission_power=specs['transmission_power']
            )
    
    def _load_complexity_levels(self):
        """بارگذاری سطوح پیچیدگی"""
        complexity_config = self.config['complexity_levels']
        
        for level_name, level_data in complexity_config.items():
            env = level_data['environment']
            tasks = level_data['tasks']
            
            self.complexity_levels[level_name] = ComplexityLevel(
                name=level_data['name'],
                num_tasks=env['num_tasks'],
                num_uavs=env['num_uavs'],
                task_size_range=tuple(tasks['data_size']),
                deadline_range=tuple(tasks['deadline'])
            )
    
    def _load_scenarios(self):
        """بارگذاری سناریوهای فعال"""
        active_scenarios = [s for s in self.config['scenarios'] if s.get('active', True)]
        
        for scenario_config in active_scenarios:
            scenario_id = scenario_config['id']
            tier = scenario_config['tier']
            complexity = scenario_config['complexity']
            
            scenario = Scenario(
                id=scenario_id,
                tier=tier,
                complexity=complexity,
                tier_specs=self.tier_specs[tier],
                complexity_specs=self.complexity_levels[complexity],
                description=f"{tier.capitalize()} + {complexity.capitalize()}"
            )
            
            self.scenarios.append(scenario)
    
    def get_scenario_by_id(self, scenario_id: str) -> Scenario:
        """دریافت سناریو با ID"""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise ValueError(f"سناریو {scenario_id} یافت نشد")
    
    def get_scenarios_by_tier(self, tier: str) -> List[Scenario]:
        """دریافت تمام سناریوهای یک Tier"""
        return [s for s in self.scenarios if s.tier == tier]
    
    def get_scenarios_by_complexity(self, complexity: str) -> List[Scenario]:
        """دریافت تمام سناریوهای یک سطح پیچیدگی"""
        return [s for s in self.scenarios if s.complexity == complexity]


if __name__ == "__main__":
    loader = ScenarioLoader()
    loader.load()
    
    print("\n📋 لیست سناریوهای بارگذاری شده:")
    for scenario in loader.scenarios:
        print(f"  - {scenario.id}: {scenario.tier} / {scenario.complexity}")
        print(f"    Tasks: {scenario.complexity_specs.num_tasks}, UAVs: {scenario.complexity_specs.num_uavs}")
