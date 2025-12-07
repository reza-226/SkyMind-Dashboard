# Computational Offloading Simulation

## Structure

- offloading_simulation/: Core simulation modules
- experiments/: Experiment scripts
- api/: Dashboard API
- results/: Results and charts

## How to Run

1. Install dependencies:
   pip install -r requirements_offloading.txt

2. Run experiments:
   python experiments/run_offloading_experiments.py

3. View results:
   python api/offloading_api.py

## Evaluation Metrics

- Scalability: Task success rate
- Energy Efficiency: Average Joules consumed
- Latency Reduction: Comparison with local processing
- Throughput: Tasks per second

## Computational Layers

1. Ground: Local processing
2. Edge: Edge server
3. Fog: Fog computing
4. Cloud: Cloud computing
