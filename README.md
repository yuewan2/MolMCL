# Multi-channel Molecular Representation Learning
This is the official code repository of the work "*Multi-channel learning for integrating structural hierarchies into context-dependent molecular representation*".

## Dependency

Follow the below steps for dependency installation.  
```
conda create -n molmcl python=3.10
conda activate molmcl
bash build.sh  # this will install all dependencies using pip
```
The main dependency of this work comes from PyTorch, PyG (torch_grometric), and RDKit. The current setup is tested under Python3.10 and CUDA11.2. The PyTorch version could be adjusted based on to your own environment. 


## Data
The data used for pre-training, including the precomputed descriptors and molecular perturbations, can be download from [here](https://drive.google.com/drive/folders/1V5OT2UF7miy4cJvqPtE7a9A6ct8GWi8D?usp=drive_link). It contains 1,861,467 data entries taken from the ZINC database, stored in the lmdb format.  

The csv files for fine-tune datasets in MoleculeNet and MoleculeACE are also included in `./data/finetune`.

### Scaffold-invariant molecular perturbations
You can also prepare your customized dataset for pre-training. The scaffold-invariant molecular perturbations can be precomputed using the following command:
```
python ./scripts/precompute.py --smiles_path <smiles_txt_file> --fragment_path <fragment_path> --save_path <save_path>
```
- `--smiles_path`: the path to the txt file which stores the SMILES sequences. An example file is included: `./data/pretrain/example.txt`.
- `--fragment_path`: the path to the fragment database used for perturbations. The fragment database used in this work can be downloaded from [here](https://www.dropbox.com/scl/fi/tezfk6odkqog1q4b3tip3/replacements02_sa2.db.gz?rlkey=iryzf7irfrjpi44cf7dag8kzf&e=1&dl=0). You can also build your own fragment database following [CReM](https://github.com/DrrDom/crem?tab=readme-ov-file).
- `--save_path`: the output directory of the computation.

This command will generate the required lmdb files for pre-training. 

## Pre-training
The configuration for pre-training, including model backbone, number of layers, and etc., can all be specified in `./config/pretrain.yaml` file. After the configuration file is setup, simply run the following command for multi-gpu training.
```
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port 12325 ./scripts/pretrain.py
```
We provide two pre-trained checkpoints (with GIN and GPS backbone) [here](https://drive.google.com/drive/folders/1G_Yejbv8LCkV5guSf1WOJq2v3Nx55e58). Put the `./checkpoint` folder in the main directory.

## Fine-tuning
The configuration files for fine-tuning MoleculeNet and MoleculeACE are located in `./config/moleculenet` and `./config/moleculeace`, respectively. The description of the available training options can be found in the yaml file. Run the command below for fine-tuning a specific dataset:
```
python ./scripts/finetune.py <data_folder>/<data_name>  # e.g., moleculenet/bace
```

## References
If you find this work useful, please cite by:

Wan, Y., Wu, J., Hou, T. et al. Multi-channel learning for integrating structural hierarchies into context-dependent molecular representation. Nat Commun 16, 413 (2025). https://doi.org/10.1038/s41467-024-55082-4

or

```
@Article{Wan2025,
         title={Multi-channel learning for integrating structural hierarchies into context-dependent molecular representation},
         author={Wan, Yue and Wu, Jialu and Hou, Tingjun and Hsieh, Chang-Yu and Jia, Xiaowei},
         journal={Nature Communications},
         year={2025}, month={Jan}, day={06},
         volume={16}, number={1}, pages={413},
         issn={2041-1723},
         doi={10.1038/s41467-024-55082-4},
         url={https://doi.org/10.1038/s41467-024-55082-4}
}
```

Thanks!
