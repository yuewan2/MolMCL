"""
Author: Derek van Tilborg -- TU/e -- 25-05-2022

Code to compute activity cliffs

    - ActivityCliffs:                   Class that takes care of computing activity cliffs
    - find_fc():                        Calculate the fold change
    - get_fc():                         Compute the pairwise fold change
    - get_levenshtein_matrix():         Compute the pairwise Levenshtein similarity
    - get_tanimoto_matrix():            Compute the pairwise Tanimoto similarity
    - get_scaffold_matrix():            Compute the pairwise scaffold similarity
    - get_mmp_matrix():                 Compute a matrix of Matched Molecular Pairs
    - mmp_similarity():                 Compute binary mmp similarity matrix
    - moleculeace_similarity():         Compute the consensus similarity (being >0.9 in at least one similarity type)

"""

from typing import List, Callable, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdMMPA import FragmentMol
from Levenshtein import distance as levenshtein
from tqdm import tqdm

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds import MurckoScaffold

RANDOM_SEED = 42


class PropertyCliffs:
    """ Activity cliff class that computes cliff compounds """
    def __init__(self, smiles: List[str], labels: Union[List[float], np.array]):
        self.smiles = smiles
        self.labels = list(labels) if type(labels) is not list else labels
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9,
                    custom_cliff_function: Callable = None,
                    mmp: bool = False,
                    reverse: bool = False):
        """ Compute activity cliffs

        :param similarity: (float) threshold value to determine structural similarity
        :param potency_fold: (float) threshold value to determine difference in bioactivity
        :param in_log10: (bool) is you bioactivty in log10 nM?
        :param custom_cliff_function: (Callable) function that takes: smiles: List[str] and similarity: float and
          returns a square binary matrix where 1 is similar and 0 is not.
        :param mmp: (bool) use matched molecular pairs to determine similarity instead
        """

        if mmp:
            sim = mmp_similarity(self.smiles)
        else:
            sim = moleculeace_similarity(self.smiles, similarity)

        if custom_cliff_function is not None:
            sim = custom_cliff_function(self.smiles, similarity)
            # sim = np.logical_or(sim == 1, custom_sim == 1).astype(int)

        self.sim = sim

        fc = np.zeros((len(self.labels), len(self.labels)))
        for i in range(len(self.labels)):
            for j in range(i + 1, len(self.labels)):
                # assume only one classification task
                if self.labels[i][0] != self.labels[j][0]:
                    fc[i][j] = 1
                    fc[j][i] = 1

        if not reverse:
            self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)
        else:
            self.cliffs = np.logical_and(sim == 1, fc == 0).astype(int)

        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        """

        :param return_smiles: (bool) return activity cliff molecules as a list of SMILES strings
        :param kwargs: arguments for ActivityCliffs.find_cliffs()
        :return: (List[int]) returns a binary list where 1 means activity cliff compounds
        """
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            return list((sum(self.cliffs) > 0).astype(int))

    def __repr__(self):
        return "Property cliffs"


class ActivityCliffs:
    """ Activity cliff class that computes cliff compounds """
    def __init__(self, smiles: List[str], bioactivity: Union[List[float], np.array]):
        self.smiles = smiles
        self.bioactivity = list(bioactivity) if type(bioactivity) is not list else bioactivity
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9, potency_fold: float = 10, in_log10: bool = True,
                    custom_cliff_function: Callable = None, mmp: bool = False, reverse: bool = False):
        """ Compute activity cliffs

        :param similarity: (float) threshold value to determine structural similarity
        :param potency_fold: (float) threshold value to determine difference in bioactivity
        :param in_log10: (bool) is you bioactivty in log10 nM?
        :param custom_cliff_function: (Callable) function that takes: smiles: List[str] and similarity: float and
          returns a square binary matrix where 1 is similar and 0 is not.
        :param mmp: (bool) use matched molecular pairs to determine similarity instead
        """

        if mmp:
            sim = mmp_similarity(self.smiles)
        else:
            sim = moleculeace_similarity(self.smiles, similarity)

        if custom_cliff_function is not None:
            sim = custom_cliff_function(self.smiles, similarity)
            # sim = np.logical_or(sim == 1, custom_sim == 1).astype(int)

        self.sim = sim

        fc = (get_fc(self.bioactivity, in_log10=in_log10) > potency_fold).astype(int)
        if not reverse:
            self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)
        else:
            self.cliffs = np.logical_and(sim == 1, fc == 0).astype(int)

        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        """

        :param return_smiles: (bool) return activity cliff molecules as a list of SMILES strings
        :param kwargs: arguments for ActivityCliffs.find_cliffs()
        :return: (List[int]) returns a binary list where 1 means activity cliff compounds
        """
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            return list((sum(self.cliffs) > 0).astype(int))

    def __repr__(self):
        return "Activity cliffs"


def find_fc(a: float, b: float):
    """Get the fold change of to bioactivities (deconvert from log10 if needed)"""

    return max([a, b]) / min([a, b])


def find_fc_explicit(a: float, b: float):
    """Get the fold change of to bioactivities (deconvert from log10 if needed)"""

    return a / b


def get_fc(bioactivity: List[float], in_log10: bool = True):
    """ Calculates the pairwise fold difference in compound activity given a list of activities"""

    bioactivity = 10 ** abs(np.array(bioactivity)) if in_log10 else bioactivity

    act_len = len(bioactivity)
    m = np.zeros([act_len, act_len])
    # Calculate upper triangle of matrix
    for i in range(act_len):
        for j in range(i, act_len):
            m[i, j] = find_fc(bioactivity[i], bioactivity[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_fc_explicit(bioactivity: List[float], in_log10: bool = True):
    """ Calculates the pairwise fold difference in compound activity given a list of activities"""

    bioactivity = 10 ** abs(np.array(bioactivity)) if in_log10 else bioactivity

    act_len = len(bioactivity)
    m = np.zeros([act_len, act_len])
    # Calculate upper triangle of matrix
    for i in range(act_len):
        for j in range(act_len):
            m[i, j] = find_fc_explicit(bioactivity[i], bioactivity[j])

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_pairwise_fc_explicit(bioactivity_1: List[float], bioactivity_2: List[float], in_log10: bool = True):
    """ Calculates the pairwise fold difference in compound activity given a list of activities"""

    bioactivity_1 = 10 ** abs(np.array(bioactivity_1)) if in_log10 else bioactivity_1
    bioactivity_2 = 10 ** abs(np.array(bioactivity_2)) if in_log10 else bioactivity_2

    m = np.zeros([len(bioactivity_1), len(bioactivity_2)])
    # Calculate upper triangle of matrix
    for i in range(len(bioactivity_1)):
        for j in range(len(bioactivity_2)):
            m[i, j] = find_fc_explicit(bioactivity_1[i], bioactivity_2[j])

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_levenshtein_matrix(smiles: List[str], normalize: bool = True):
    """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string"""

    smi_len = len(smiles)

    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in (range(smi_len)):
        for j in range(i, smi_len):
            if normalize:
                m[i, j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            else:
                m[i, j] = levenshtein(smiles[i], smiles[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Get from a distance to a similarity
    m = 1 - m

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_scaffold(smi, generic=False):
    mol = Chem.MolFromSmiles(smi)
    if generic:
        return Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)


def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, use_scaffold=False):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles:
        if use_scaffold:
            m = Chem.MolFromSmiles(get_scaffold(smi))
        else:
            m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in (range(smi_len)):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]],
                                                     db_fp[smiles[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_scaffold_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string """

    # Make scaffold database
    db_scaf = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(m)
        except Exception:  # In the very rare case this doesn't work, use a normal scaffold
            print(f"Could not create a generic scaffold of {smi}, used a normal scaffold instead")
            skeleton = GetScaffoldForMol(m)
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        db_scaf[smi] = skeleton_fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in (range(smi_len)):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]],
                                                     db_scaf[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def find_fragments(smiles: List[str]):
    """ Build a database of molecular fragments for matched molecular pair analysis. Molecular fragmentation from the
    Hussain and Rae algorithm is used. We only use a single cut (true to the original MMP idea) """

    db = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        cuts = FragmentMol(m, maxCuts=1, resultsAsMols=False)

        # extract all not None fragments into a flat list
        fragments = sum([[i for i in cut if i != ''] for cut in cuts], [])

        # Keep the largest fragment as the core structure.
        for i, frag in enumerate(fragments):
            split_frags = frag.split('.')
            if Chem.MolFromSmiles(split_frags[0]).GetNumAtoms() >= Chem.MolFromSmiles(split_frags[-1]).GetNumAtoms():
                core = split_frags[0]
            else:
                core = split_frags[-1]

            # Ignore dummy variables for matching (otherwise you will never get a match). Dummies are introduced during
            # fragmentation
            qp = Chem.AdjustQueryParameters()
            qp.makeDummiesQueries = True
            qp.adjustDegreeFlags = Chem.ADJUST_IGNOREDUMMIES
            fragments[i] = Chem.AdjustQueryProperties(Chem.MolFromSmiles(core), qp)
        # Add all core fragments to the dictionary
        db[smi] = fragments

    return db


def mmp_match(smiles: str, fragments: List):
    """ Check if fragments (provided as rdkit Mol objects are substructures of another molecule (from SMILES string) """

    m = Chem.MolFromSmiles(smiles)
    for frag in fragments:
        # Match fragment on molecule
        if m.HasSubstructMatch(frag):
            return 1
    return 0


def get_mmp_matrix(smiles: List[str]):
    """ Calculates a matrix of matched molecular pairs for a list of SMILES string"""

    # Make a fingerprint database
    db_frags = find_fragments(smiles)

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix.
    for i in (range(smi_len)):
        for j in range(i, smi_len):
            m[i, j] = mmp_match(smiles[i], db_frags[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def mmp_similarity(smiles: List[str], similarity=None):
    """ Calculate which pairs of molecules are matched molecular pairs """

    return (get_mmp_matrix(smiles) > 0).astype(int)


def tanimoto_similarity(smiles: List[str], similarity: float = 0.9):
    m_tani = get_tanimoto_matrix(smiles) >= similarity
    return m_tani.astype(int)


def moleculeace_similarity(smiles: List[str], similarity: float = 0.9):
    """ Calculate which pairs of molecules have a high tanimoto, scaffold, or SMILES similarity """

    m_tani = get_tanimoto_matrix(smiles) >= similarity
    m_scaff = get_scaffold_matrix(smiles) >= similarity
    m_leve = get_levenshtein_matrix(smiles) >= similarity

    return (m_tani + m_scaff + m_leve).astype(int)


def moleculeace_pairwise_similarity(smiles_1: List[str], smiles_2: List[str], similarity: float = 0.9):
    """ Calculate which pairs of molecules have a high tanimoto, scaffold, or SMILES similarity """

    m_tani = get_pairwise_tanimoto_matrix(smiles_1, smiles_2) >= similarity
    m_scaff = get_pairwise_scaffold_matrix(smiles_1, smiles_2) >= similarity
    m_leve = get_pairwise_levenshtein_matrix(smiles_1, smiles_2) >= similarity

    return (m_tani + m_scaff + m_leve).astype(int)


def get_pairwise_tanimoto_matrix(smiles_1: List[str], smiles_2: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles_1 + smiles_2:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    m = np.zeros([len(smiles_1), len(smiles_2)])
    # Calculate upper triangle of matrix
    for i in range(len(smiles_1)):
        for j in range(len(smiles_2)):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles_1[i]],
                                                     db_fp[smiles_2[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    # m = m + m.T - np.diag(np.diag(m))
    # # Fill the diagonal with 0's
    # np.fill_diagonal(m, 0)

    return m


def get_pairwise_scaffold_matrix(smiles_1: List[str], smiles_2: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string """

    # Make scaffold database
    db_scaf = {}
    for smi in smiles_1+smiles_2:
        m = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(m)
        except Exception:  # In the very rare case this doesn't work, use a normal scaffold
            # print(f"Could not create a generic scaffold of {smi}, used a normal scaffold instead")
            skeleton = GetScaffoldForMol(m)
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        db_scaf[smi] = skeleton_fp

    m = np.zeros([len(smiles_1), len(smiles_2)])
    # Calculate upper triangle of matrix
    for i in (range(len(smiles_1))):
        for j in range(len(smiles_2)):
            m[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles_1[i]],
                                                     db_scaf[smiles_2[j]])

    # # Fill in the lower triangle without having to loop (saves ~50% of time)
    # m = m + m.T - np.diag(np.diag(m))
    # # Fill the diagonal with 0's
    # np.fill_diagonal(m, 0)

    return m


def get_pairwise_levenshtein_matrix(smiles_1: List[str], smiles_2: List[str], normalize: bool = True):
    """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string"""

    m = np.zeros([len(smiles_1), len(smiles_2)])
    # Calculate upper triangle of matrix
    for i in (range(len(smiles_1))):
        for j in range(len(smiles_2)):
            if normalize:
                m[i, j] = levenshtein(smiles_1[i], smiles_2[j]) / max(len(smiles_1[i]), len(smiles_2[j]))
            else:
                m[i, j] = levenshtein(smiles_1[i], smiles_2[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    # m = m + m.T - np.diag(np.diag(m))
    # # Get from a distance to a similarity
    # m = 1 - m
    #
    # # Fill the diagonal with 0's
    # np.fill_diagonal(m, 0)

    return m