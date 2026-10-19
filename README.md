# cosmosis
A datascience framework and tools for rapid prototyping.

## Datasets, models and learners implemented in pytorch
* light weight, modular, extendable
* pytorch=2.14, pytorch-cuda=12.6, python=3.12
* jupyterlab, numpy, pandas, scipy, scikit-learn, matplotlib, torcheval

## instructions for setting up repo and environment
git clone https://github.com/icanswim/cosmosis.git
cd cosmosis

## recommend uv environment
uv sync
uv run jupyter lab

## conda environment
conda create --name cosmo python=3.12 pip -y
conda activate cosmo
pip install -r requirements-dev.txt
pip install torcheval
jupyter lab

## manual creation of conda enviroment
conda create --name cosmo python=3.12
conda activate cosmo
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
conda install -c conda-forge scipy scikit-learn
conda install -c conda-forge numpy
conda install -c conda-forge pandas matplotlib
conda install -c conda-forge juptyerlab
pip install torcheval
jupyter lab





