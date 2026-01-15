# RailPricing-RL: Multi-Agent Reinforcement Learning for Railway Pricing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ArXiv](https://img.shields.io/badge/arXiv-2501.08234-b31b1b.svg)](https://doi.org/10.48550/arXiv.2501.08234)

## 🛠️ Installation

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kinrre/RailPricing-RL.git
   cd RailPricing-RL

2. **Install dependencies and project in editable mode:**
    ```bash
    pip3 install -r requirements.txt
    pip3 install -e .

## 💻 Usage

1. **Training the Agent:**

    To train a new RL agent from scratch, run the following command:
    ```bash
    python3 -m tests.test_rl_training --supply-config configs/rl/supply_data_connecting.yml --demand-config configs/rl/demand_data_student.yml --seed 0 --exp-name business_student --algorithm iql_sac --total-timesteps 1400000

2. **Evaluation:**

    ```bash
    python3 -m tests.test_rl_evaluator --seed 0 --input_dir $input_dir --algorithm iql_sac --total-timesteps 70000

## 📜 Citation

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@article{villarrubia2025dynamic,
  title={Dynamic Pricing in High-Speed Railways Using Multi-Agent Reinforcement Learning},
  author={Villarrubia-Martin, Enrique Adrian and Rodriguez-Benitez, Luis and Mu{\~n}oz-Valero, David and Montana, Giovanni and Jimenez-Linares, Luis},
  journal={arXiv preprint arXiv:2501.08234},
  year={2025}
}
