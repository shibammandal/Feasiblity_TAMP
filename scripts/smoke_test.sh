#!/bin/bash
set -e
echo "1. Generating Data..."
.venv/bin/python scripts/generate_data.py --num_samples 10 --output data_test --actions_per_scene 2

echo "2. Training Model (MLP)..."
.venv/bin/python scripts/train_model.py --model mlp --epochs 1 --data data_test/feasibility_dataset.h5 --batch_size 4 --checkpoint_dir checkpoints_test --log_dir logs_test

echo "3. Running Benchmark..."
.venv/bin/python scripts/run_benchmark.py --num_scenarios 2 --checkpoint checkpoints_test/best_model.pt --data data_test/feasibility_dataset.h5 --output benchmark_test.json --quick

echo "Smoke Test Complete!"
