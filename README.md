# battery-capacity-mlp-baseline
## Overview
This repository contains an initial machine learning model for estimating battery capacity. It implements a simple **Multilayer Perceptron** to establish a baseline for capacity degradation prediction.

The primary objective is to:
1. Validate existing data preprocessing pipelines.
2. Feature engineer signals that can serve as features to model degradation at different Depths of Discharge (DoD).
3. Provide a low-complexity baseline for comparison against future temporal architectures.

## Methodology
Since the dataset is lacking several useful features, we use feature engineering to build a model that can depict battery degradation across different DoDs, using terminal voltage as a proxy for SoC, allowing the model to infer effective DoD levels.

The model utilizes features such as:
1. The rate of voltage decline $(\Delta V / \Delta t)$.
2. Time elapsed during each segment.
3. Average current during each interval.
4. Explicit $v_{start}$ and $v_{end}$ values.
5. The cycle number.
6. Ambient Temperature.

## Requirements
To replicate the environment, install the dependencies:

```bash
pip install -r requirements.txt
