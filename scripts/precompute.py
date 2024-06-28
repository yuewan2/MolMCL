import os
import copy
import lmdb
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from functools import partial
from crem.crem import mutate_mol2  # Get perturbations by crem
from multiprocessing import Pool

from molmcl.utils.data import get_scaffold
from molmcl.utils.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

generator = RDKit2DNormalized()


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

        mapped_inds = mol.GetSubstructMatch(Chem.MolFromSmiles(scaff_smi))
        replace_inds = set(np.arange(len(mol.GetAtoms()))) - set(mapped_inds)
        for idx in copy.deepcopy(replace_inds):
            for n_atom in mol.GetAtomWithIdx(int(idx)).GetNeighbors():
                replace_inds.add(n_atom.GetIdx())
        replace_ids = [int(i) for i in replace_inds]

        # Mutate molecule:
        random_mutations = set()
        if replace_inds:
            for radius in [2, 3]:
                random_mutations.update(set(mutate_mol2(mol, db_name=db_name,
                                                        max_size=5, radius=radius,
                                                        replace_ids=replace_ids)))
            mutations = filter_by_scaffold(scaff_smi, list(random_mutations), identical=True, max_amount=max_amount)
        else:
            mutations = []

    output = {'id': smi_id,
              'smi': smi,
              'scaff_smi': scaff_smi,
              'md': list(md),
              'md_scaff': list(scaff_md),
              'mutate': mutations}

    return output


def main(args):
    ids, smiles, is_perturb_valid = [], [], []
    with open(args.smiles_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            smiles.append(line.strip())
            ids.append('z{}'.format(i))

    key2shard_id = {}
    chunk_size = 10
    for shard_id in tqdm(range(len(smiles)//chunk_size + 1)):
        smiles_shard = smiles[shard_id*chunk_size:(shard_id+1)*chunk_size]

        os.makedirs(os.path.join(args.save_dir, 'precompute'), exist_ok=True)
        env = lmdb.open(os.path.join(args.save_dir, 'precompute', '{}.lmdb'.format(shard_id)), map_size=1099511627776)

        with env.begin(write=True) as txn:
            pool = Pool()
            for output in tqdm(pool.imap_unordered(partial(compute, db_name=args.fragment_path),
                                                   zip(smiles_shard, ids)), total=len(smiles_shard)):
                if output:
                    key = output.pop('id').encode()
                    try:
                        txn.put(key, json.dumps(output).encode())
                        key2shard_id[key.decode()] = shard_id   
                    except:
                        print('[warning] ignoring smiles {}.'.format(key.decode()))

    with open(os.path.join(args.save_dir, 'precompute/data_shard_ids.pkl'), 'wb') as f:
        pickle.dump(key2shard_id, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_path', type=str,
                        help='path of .txt file where each line corresponds a SMILES sequence')
    parser.add_argument('--fragment_path', type=str, help='path of fragment database')
    parser.add_argument('--save_dir', type=str, help='directory to save computed results')
    args = parser.parse_args()
    # example call:
    # python ./scripts/precompute.py --smiles_path ./data/pretrain/example.txt --fragment_path /data/yuw253/replacements02_sa2.db --save_dir ./data/pretrain
    main(args)