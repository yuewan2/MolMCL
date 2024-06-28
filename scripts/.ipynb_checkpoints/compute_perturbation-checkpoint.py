import os
import copy
import lmdb
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from crem.crem import mutate_mol2  # Get perturbations by crem
from multiprocessing import Pool


from molmcl.utils.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

generator = RDKit2DNormalized()


parser = argparse.ArgumentParser()
parser.add_argument('--smiles_path', type=str, help='path of .txt file where each line corresponds a SMILES sequence')
parser.add_argument('--fragment_path', type=str, help='path of fragment database')
parser.add_argument('--save_path', type=str, help='path to save computed results')
args = parser.parse_args()


def filter_by_scaffold(anchor_scaffold, perturbations, identical=True, max_amount=100):
    np.random.shuffle(perturbations)

    filtered_perturbations = []
    for perturb in perturbations:
        p_scaffold = get_scaffold(perturb)
        if identical and anchor_scaffold == p_scaffold:
            filtered_perturbations.append(perturb)
        elif not identical and anchor_scaffold != p_scaffold:
            filtered_perturbations.append(perturb)

        if len(filtered_perturbations) == max_amount:
            break

    return filtered_perturbations


def compute(task, db_name, max_amount=100, max_atom=200):
    smi, smi_id = task

    # Get mutation indices
    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > max_atom:  
        return {}

    md = np.array(list(generator.process(smi)), dtype=float)[1:]

    scaff_smi = get_scaffold(mol)
    if not scaff_smi:
        scaff_md = np.zeros(200, dtype=float)
        mutations, random_mutations = [], []
    else:
        scaff_md = np.array(list(generator.process(scaff_smi)), dtype=float)[1:]

        # Mutate molecule:
        try:
            random_mutations = list(set(mutate_mol2(mol, db_name=db_name, max_size=5, radius=2)))
            mutations = filter_by_scaffold(scaff_smi, random_mutations, identical=True, max_amount=max_amount)
        except:
            random_mutations = []
            mutations = []

        if len(mutations) > max_amount:
            mutations = list(np.random.choice(mutations, size=max_amount, replace=False))
        if len(random_mutations) > max_amount:
            random_mutations = list(np.random.choice(random_mutations, size=max_amount, replace=False))

    output = {'id': smi_id,
              'smi': smi,
              'scaff_smi': scaff_smi,
              'md': list(md),
              'md_scaff': list(scaff_md),
              'mutate': mutations}

    return output


def main(input_path, db_path, save_dir):
    
    ids, smiles, is_perturb_valid = [], [], []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            smiles.append(line.strip())
            ids.append('z{}'.format(i))

    # smiles = smiles[:10]
    # ids = ids[:10]

    key2shard_id = {}
    chunk_size=10000
    visited = set()
    
    for shard_id in tqdm(range(len(smiles)//chunk_size + 1)):
        smiles_shard = smiles[shard_id*chunk_size:(shard_id+1)*chunk_size]

        with env.begin(write=True) as txn:
            pool = Pool()
            for output in tqdm(pool.imap_unordered(partial(compute, db_name=db_path), zip(smiles_shard, ids)), total=len(smiles_shard)):
                if output:
                    key = output.pop('id').encode()
                    try:
                        txn.put(key, json.dumps(output).encode())
                        key2shard_id[key.decode()] = shard_id   
                    except:
                        print('[warning] ignoring smiles {}.'.format(key.decode()))



if __name__ == '__main__':
    input_path = './data/pretrain/chembl.txt'
    db_name = './data/replacements02_sa2.db'
    save_dir = './data/pretrain'
    
    input_path, db_path, save_dir
    
    main('/data/yuw253/mol_pretrain')