# Behavioral Cloning Agent

A machine learning project implementing an imitation learning (behavioral cloning) agent trained with PyTorch. The model learns to predict actions directly from recorded environment observations, mimicking demonstrated behavior rather than learning through trial-and-error reinforcement.

## Overview

- **Data collection**: `record.py` captures observation-action pairs from the environment, saved as `observations.npy` and `actions.npy`.
- **Training**: `training.py` and `training3sloja.py` train the behavioral cloning model (the latter using a deeper, 3-layer network architecture).
- **Testing**: `test.py` and `test3sloja.py` evaluate the trained model's performance.
- **Model**: `bc_model.pth` — the trained PyTorch model checkpoint.
- **Notebooks**: `1.ipynb` and `2d2.ipynb` contain exploratory analysis and experimentation.

## Tech Stack

- Python
- PyTorch
- NumPy

## How it works

1. Demonstrations (observations and corresponding actions) are recorded from the environment.
2. A neural network is trained via supervised learning to map observations to actions, minimizing the difference between predicted and demonstrated actions.
3. The trained model is evaluated by running it in the environment and observing its behavior.

## Setup

```bash
pip install -r requirements.txt
python training.py
python test.py
```
