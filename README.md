

# LeWorldModel Scientific Computing Project

This repository contains our Scientific Computing project based on **LeWorldModel (LeWM)**, a stable end-to-end latent world model trained from pixel observations. The project studies how LeWM learns compact latent dynamics, performs planning in latent space, and evaluates control performance across multiple environments.

## Project Overview

LeWorldModel is a Joint-Embedding Predictive Architecture (JEPA) for learning world models from raw visual observations. Instead of predicting future pixels directly, the model encodes observations into a latent space and predicts future latent states conditioned on actions.

The main ideas covered in this project are:

- Learning latent dynamics from offline trajectories
- Avoiding representation collapse with SIGReg regularization
- Performing planning with Model Predictive Control
- Evaluating success rates across control tasks
- Analyzing computational efficiency and reproducibility

## Evaluation Results

The following results were recorded from 50 evaluation episodes per task:

| Task | Successful Episodes | Success Rate | Evaluation Time |
|---|---:|---:|---:|
| Push-T | 47 / 50 | 94.0% | 128.06 s |
| DMC | 40 / 50 | 80.0% | 171.33 s |
| Two-Room | 43 / 50 | 86.0% | 202.37 s |



## Notes on Reproducibility

The provided evaluation logs report `seeds: None`, so the current results should be interpreted as recorded evaluation runs rather than fully seeded reproducibility benchmarks. For stronger reproducibility, future runs should record:

- random seed
- model checkpoint
- environment version
- number of episodes
- planning horizon
- optimizer and training settings
- hardware used for evaluation

## Conclusion

This project demonstrates how latent world models can reduce the computational cost of planning while preserving task-relevant dynamics. LeWorldModel is especially useful as a scientific computing case study because it combines representation learning, stochastic optimization, random projection methods, and model predictive control.

## Original Paper
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}"# LWM_ScientificComputing" 
"# LWM_ScientificComputing" 
