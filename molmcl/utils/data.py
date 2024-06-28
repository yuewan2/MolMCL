import copy
import json
import os
import torch
import pickle
import collections
from collections import Counter
import math
import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric, MurckoScaffoldSmiles

from torch.utils import data
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import to_networkx
from torch_geometric.transforms.virtual_node import VirtualNode


hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
hydrogen_acceptor = Chem.MolFromSmarts(
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
    "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
basic = Chem.MolFromSmarts(
    "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
    "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


get_fp = lambda mol: AllChem.GetMorganFingerprintAsBitVect(
    mol, radius=2, nBits=512)


def get_scaffold(smi, generic=False, return_mol=False):
    if generic:
        assert not return_mol

    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = smi

    if generic:
        if return_mol:
            return MakeScaffoldGeneric(mol)
        else:
            return Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        if return_mol:
            return GetScaffoldForMol(mol)
        else:
            return Chem.MolToSmiles(GetScaffoldForMol(mol))


def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def mol_to_graph_data_obj_super_rich(mol, virtual_node=False):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses full atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """

    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    ring_info = mol.GetRingInfo()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        feature = onek_encoding_unk(atom.GetAtomicNum(), list(range(119))) + \
                  onek_encoding_unk(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) + \
                  onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + \
                  onek_encoding_unk(atom.GetChiralTag(), [0, 1, 2, 3]) + \
                  onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + \
                  onek_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                              Chem.rdchem.HybridizationType.SP2,
                                                              Chem.rdchem.HybridizationType.SP3,
                                                              Chem.rdchem.HybridizationType.SP3D,
                                                              Chem.rdchem.HybridizationType.SP3D2]) + \
                  [1 if atom.GetIsAromatic() else 0] + \
                  [atom.GetMass() * 0.01] + \
                  onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom_idx in hydrogen_acceptor_match] + \
                  [atom_idx in hydrogen_donor_match] + \
                  [atom_idx in acidic_match] + \
                  [atom_idx in basic_match] + \
                  [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                   ring_info.IsAtomInRingOfSize(atom_idx, 4),
                   ring_info.IsAtomInRingOfSize(atom_idx, 5),
                   ring_info.IsAtomInRingOfSize(atom_idx, 6),
                   ring_info.IsAtomInRingOfSize(atom_idx, 7),
                   ring_info.IsAtomInRingOfSize(atom_idx, 8)]

        atom_features_list.append(feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 14  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feature = onek_encoding_unk(bond.GetBondType(), allowable_features['possible_bonds']) + \
                      onek_encoding_unk(int(bond.GetStereo()), list(range(6))) + \
                      [bond.GetIsConjugated()] + \
                      [bond.IsInRing()]

            edges_list.append((i, j))
            edge_features_list.append(feature)
            edges_list.append((j, i))
            edge_features_list.append(feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if virtual_node:
        data = VirtualNode()(data)

    return data


def mol_to_graph_data_obj_rich(mol, virtual_node=False):
    # num_atom_features = 143
    atom_features_list = []
    for atom in mol.GetAtoms():
        feature = onek_encoding_unk(atom.GetAtomicNum(), list(range(1, 101)))
        feature += onek_encoding_unk(atom.GetDegree(), list(range(11)))
        feature += onek_encoding_unk(atom.GetFormalCharge(), list(range(-2, 3)))
        feature += onek_encoding_unk(atom.GetNumRadicalElectrons(), list(range(5)))
        feature += onek_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                               Chem.rdchem.HybridizationType.SP2,
                                                               Chem.rdchem.HybridizationType.SP3,
                                                               Chem.rdchem.HybridizationType.SP3D,
                                                               Chem.rdchem.HybridizationType.SP3D2])
        feature += [1 if atom.GetIsAromatic() else 0]
        feature += onek_encoding_unk(atom.GetTotalNumHs(), list(range(5)))
        feature += [1 if atom.HasProp('_ChiralityPossible') else 0]
        if not atom.HasProp('_CIPCode'):
            feature += [0, 0, 0]
        else:
            feature += onek_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S'])
        feature += [atom.GetMass() * 0.01]
        atom_features_list.append(feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 14
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            feature = onek_encoding_unk(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                                             Chem.rdchem.BondType.DOUBLE,
                                                             Chem.rdchem.BondType.TRIPLE,
                                                             Chem.rdchem.BondType.AROMATIC])
            feature += [1 if bond.GetIsConjugated() else 0]
            feature += [1 if bond.IsInRing() else 0]
            feature += onek_encoding_unk(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREONONE,
                                                            Chem.rdchem.BondStereo.STEREOANY,
                                                            Chem.rdchem.BondStereo.STEREOZ,
                                                            Chem.rdchem.BondStereo.STEREOE,
                                                            Chem.rdchem.BondStereo.STEREOCIS,
                                                            Chem.rdchem.BondStereo.STEREOTRANS])
            edges_list.append((i, j))
            edge_features_list.append(feature)
            edges_list.append((j, i))
            edge_features_list.append(feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if virtual_node:
        data = VirtualNode()(data)

    return data


def mol_to_graph_data_obj_basic(mol, virtual_node=False):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if virtual_node:
        data = VirtualNode()(data)

    return data


def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :param: data_edge_attr
    :return:
    """
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol


def list_collate_fn(data_list):
    num_candidates = len(data_list[0])

    data_full = []
    for d in data_list:
        data_full.extend(d)

    batch = Batch.from_data_list(data_full)

    return batch


def list_collate_fn_pretrain(data_list):
    num_candidates = len(data_list[0])
    print(num_candidates)
    print(len(data_list))

    data_full = []
    data_anchor = []
    for d in data_list:
        data_full.extend(d)
        data_anchor.append(d[0])

    batch = Batch.from_data_list(data_full)

    return batch, data_anchor


def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output


def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output
