# Evaluating Key Design Aspects in Simulated Hybrid Quantum Neural Networks for Earth Observation

This repository contains code and resources to explore key design aspects of Hybrid Quantum Neural Networks (HQNN) applied to Earth Observation tasks. The work examines performance factors, comparing various model architectures using simulated quantum environments.

## Getting Started

To run the code, you can use the following Docker images:

- **For Qiskit models**: `lorenzopapa5/cuda12.1.0-python3.8-pytorch2.4.0-esa`
- **For PennyLane models**: `lorenzopapa5/cuda12.1.0-python3.8-pytorch2.4.0-esa-hqm`

These images include all necessary dependencies, including CUDA, Python 3.8, PyTorch 2.4.0, and respective quantum libraries for the models used in this study.

### Running the Code

- **To run Qiskit models**, execute: `main_binary_qkit.py`
- **To run PennyLane models**, execute: `main_binary_pl.py`

Ensure you are using the correct Docker image depending on the model you are running.

## Paper

For more details on the design aspects and methodologies used in this study, please refer to our [arXiv paper](https://arxiv.org/abs/2410.08677).

## Citation

If you use this code or reference our work, please cite our paper as follows:

```bibtex
@article{papa2024impact,
  title={On the impact of key design aspects in simulated Hybrid Quantum Neural Networks for Earth Observation},
  author={Papa, Lorenzo and Sebastianelli, Alessandro and Meoni, Gabriele and Amerini, Irene},
  journal={arXiv preprint arXiv:2410.08677},
  year={2024}
}
