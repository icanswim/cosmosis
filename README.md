# cosmosis

A data science framework and tools for rapid prototyping.

## Datasets, Models, and Learners Implemented in PyTorch

* Light-weight, modular, and extendable.
* **Core Stack:** `pytorch=2.14`, `pytorch-cuda=12.6`, `python=3.12`
* **Dependencies:** `jupyterlab`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `torcheval`

---

## Instructions for Setting Up Repo and Environment

```bash
git clone https://github.com/icanswim/cosmosis.git
cd cosmosis
```

### Recommended: `uv` Environment

```bash
uv sync
uv run jupyter lab
```

### Conda Environment (Automated)

```bash
conda create --name cosmo python=3.12 pip -y
conda activate cosmo
pip install -r requirements-dev.txt
pip install torcheval
jupyter lab
```

### Conda Environment (Manual Creation)

```bash
conda create --name cosmo python=3.12
conda activate cosmo
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
conda install -c conda-forge scipy scikit-learn
conda install -c conda-forge numpy
conda install -c conda-forge pandas matplotlib
conda install -c conda-forge jupyterlab
pip install torcheval
jupyter lab
```

---

## Repository Structure

```text
.
├── data
│   ├── cifar-10-batches-py
│   └── MNIST
│       └── raw
├── log
├── model
├── __pycache__
└── src
    └── cosmosis
```

---

## Installed Packages Reference

| Package | Version |
| :--- | :--- |
| anyio | 4.15.1 |
| argon2-cffi | 25.1.0 |
| argon2-cffi-bindings | 26.1.0 |
| arrow | 1.4.0 |
| asttokens | 3.0.2 |
| async-lru | 2.4.0 |
| attrs | 26.1.0 |
| babel | 2.18.0 |
| beautifulsoup4 | 4.15.0 |
| bleach | 6.4.0 |
| certifi | 2026.10.18 |
| cffi | 2.1.1 |
| charset-normalizer | 3.5.2 |
| cloudpickle | 3.1.2 |
| comm | 0.2.3 |
| contourpy | 1.4.0 |
| cuda-bindings | 12.9.10 |
| cuda-pathfinder | 1.8.3 |
| cuda-toolkit | 12.6.3 |
| cycler | 0.12.1 |
| debugpy | 1.8.22 |
| defusedxml | 0.7.1 |
| executing | 2.3.0 |
| fastjsonschema | 2.22.2 |