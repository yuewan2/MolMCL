# deep learning dependency
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric

# cheminformatics dependency
pip install rdkit
pip install rogi  # for roughness index QSPR metric (https://github.com/coleygroup/rogi)
pip install crem  # for molecule perturbation (https://github.com/DrrDom/crem) (optional)

# general dependency
pip install matplotlib
pip install tqdm
pip install lmdb
pip install levenshtein
pip install cairosvg
pip install pandas
pip install yaml

pip install -e .

# prepare default folder
mkdir checkpoint

