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

### Conda Environment

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

## Installed Packages

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
| filelock | 4.2.4 |
| fonttools | 4.67.0 |
| fqdn | 1.6.0 |
| fsspec | 2026.9.0 |
| h11 | 0.16.0 |
| httpcore | 1.0.9 |
| httpx | 0.28.1 |
| idna | 3.20 |
| ipykernel | 7.4.0 |
| ipython | 9.17.1 |
| ipython-pygments-lexers | 1.1.1 |
| isoduration | 20.11.0 |
| jedi | 0.20.1 |
| jinja2 | 3.1.6 |
| joblib | 1.6.0 |
| json5 | 0.17.3 |
| jsonpointer | 3.4 |
| jsonschema | 4.26.0 |
| jsonschema-specifications | 2025.9.1 |
| jupyter-builder | 1.2.3 |
| jupyter-client | 8.10.0 |
| jupyter-core | 5.9.1 |
| jupyter-events | 0.12.1 |
| jupyter-lsp | 2.3.1 |
| jupyter-server | 2.21.1 |
| jupyter-server-terminals | 0.5.4 |
| jupyterlab | 4.6.4 |
| jupyterlab-pygments | 0.3.0 |
| jupyterlab-server | 2.28.1 |
| kiwisolver | 1.5.1 |
| lark | 1.3.1 |
| markupsafe | 3.0.4 |
| matplotlib | 3.11.2 |
| matplotlib-inline | 0.2.2 |
| mistune | 3.4.0 |
| mpmath | 1.3.0 |
| narwhals | 2.27.1 |
| nbclient | 0.11.0 |
| nbconvert | 7.17.2 |
| nbformat | 5.11.1 |
| nest-asyncio2 | 1.7.4 |
| networkx | 3.7 |
| notebook-shim | 0.2.4 |
| numpy | 2.5.4 |
| nvidia-cublas-cu12 | 12.6.4.1 |
| nvidia-cuda-cupti-cu12 | 12.6.80 |
| nvidia-cuda-nvrtc-cu12 | 12.6.85 |
| nvidia-cuda-runtime-cu12 | 12.6.77 |
| nvidia-cudnn-cu12 | 9.10.2.21 |
| nvidia-cufft-cu12 | 11.3.0.4 |
| nvidia-cufile-cu12 | 1.11.1.6 |
| nvidia-curand-cu12 | 10.3.7.77 |
| nvidia-cusolver-cu12 | 11.7.1.2 |
| nvidia-cusparse-cu12 | 12.5.4.2 |
| nvidia-cusparselt-cu12 | 0.7.1 |
| nvidia-nccl-cu12 | 2.29.3 |
| nvidia-nvjitlink-cu12 | 12.9.86 |
| nvidia-nvshmem-cu12 | 3.4.5 |
| nvidia-nvtx-cu12 | 12.6.77 |
| packaging | 26.3 |
| pandas | 3.0.6 |
| pandocfilters | 1.5.1 |
| parso | 0.8.7 |
| pexpect | 4.9.0 |
| pillow | 13.0.0 |
| platformdirs | 4.13.5 |
| prometheus-client | 0.26.0 |
| prompt-toolkit | 3.0.53 |
| psutil | 7.2.2 |
| ptyprocess | 0.7.0 |
| pure-eval | 0.2.4 |
| pycparser | 3.11 |
| pygments | 2.21.0 |
| pyparsing | 3.3.3 |
| python-dateutil | 2.9.0.post0 |
| python-json-logger | 4.2.0 |
| pyyaml | 6.0.3 |
| pyzmq | 27.2.0 |
| referencing | 0.37.0 |
| requests | 2.34.2 |
| rfc3339-validator | 0.1.4 |
| rfc3986-validator | 0.1.1 |
| rfc3987-syntax | 1.1.0 |
| rpds-py | 2026.9.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| send2trash | 2.1.0 |
| setuptools | 84.0.0 |
| six | 1.17.0 |
| soupsieve | 3.0.3 |
| stack-data | 0.6.3 |
| sympy | 1.14.0 |
| terminado | 0.18.1 |
| threadpoolctl | 3.7.0 |
| tinycss2 | 1.5.1 |
| torch | 2.14.1+cu126 |
| torchaudio | 2.11.0+cu126 |
| torcheval | 0.0.7 |
| torchvision | 0.29.1+cu126 |
| tornado | 6.5.10 |
| traitlets | 5.16.1 |
| triton | 3.8.0 |
| typing-extensions | 4.16.0 |
| tzdata | 2026.5 |
| uri-template | 1.3.0 |
| urllib3 | 2.8.0 |
| wcwidth | 0.9.2 |
| webcolors | 25.10.0 |
| webencodings | 0.6.1 |
| websocket-client | 1.9.2 |

