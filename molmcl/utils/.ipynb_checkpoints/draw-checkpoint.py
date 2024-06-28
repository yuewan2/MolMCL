import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from scipy.interpolate import griddata
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

from cairosvg import svg2png
from IPython.display import SVG, display


def draw_rxn(gt, size=(900, 300), highlight=False):
    rxn = AllChem.ReactionFromSmarts(gt)
    d = Draw.MolDraw2DSVG(size[0], size[1])
    colors = [(1, 0.6, 0.6), (0.4, 0.6, 1)]

    d.DrawReaction(rxn, highlightByReactant=highlight, highlightColorsReactants=colors)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    svg2 = svg.replace('svg:', '')
    svg3 = SVG(svg2)
    display(svg3)


def get_pair(atoms):
    atom_pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            atom_pairs.append((atoms[i], atoms[j]))
    return atom_pairs


def get_bonds(mol, atoms):
    atom_pairs = get_pair(atoms)
    bond_list = []
    for ap in atom_pairs:
        bond = mol.GetBondBetweenAtoms(*ap)
        if bond is not None:
            bond_list.append(bond.GetIdx())
    return list(set(bond_list))


def draw_mols_with_fr(smiles, smarts_list, size=(300, 100), save_path='', color_map=None, color_constant='lightcoral', straighten=False):
    mol_list, matched_atom_list, noise_matched_atom_list = [], [], []
    matched_bond_list, noise_matched_bond_list = [], []

    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles  
        
    for i in range(len(smarts_list)):    
        patt = Chem.MolFromSmarts(smarts_list[i])
        atoms_matched = mol.GetSubstructMatch(patt)
        bonds_matched = get_bonds(mol, atoms_matched)
        matched_atom_list.append(atoms_matched)
        matched_bond_list.append(bonds_matched)

    all_matched_atom_list, all_matched_bond_list = set(), set()
    all_matched_atom_color, all_matched_bond_color = {}, {}

    for i in range(len(matched_atom_list)):

        if color_map is None:
            color1 = matplotlib.colors.to_rgb(color_constant)
        else:
            color1 = color_map(i)
        
        all_matched_atom_list.update(matched_atom_list[i])
        all_matched_bond_list.update(matched_bond_list[i])
        atom2color = {a: color1 for a in matched_atom_list[i]}
        bond2color = {b: color1 for b in matched_bond_list[i]}

        all_matched_atom_color.update(atom2color)
        all_matched_bond_color.update(bond2color)
        
    if straighten:
        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)

    svg = Draw.MolsToGridImage([mol], subImgSize=size, useSVG=True, molsPerRow=2,
                               highlightAtomLists=[list(all_matched_atom_list)],
                               highlightAtomColors=[all_matched_atom_color],
                               highlightBondLists=[list(all_matched_bond_list)],
                               highlightBondColors=[all_matched_bond_color], maxMols=99999)
    # svg = SVG(svg)
    display(svg)
    if save_path:
        svg2png(bytestring=svg.data, write_to=save_path, dpi=300)


def draw_mols(smiles_list, smarts_list=None, noise_smarts_list=None, highlight=False, anchor_smi=None,
              size=(300, 100), save_path='', color=None, straighten=False):
    if color is None:
        color1 = matplotlib.colors.to_rgb('lightcoral')
    else:
        color1 = color
    color2 = matplotlib.colors.to_rgb('cornflowerblue')  # cornflowerblue; darkseagreen
    mol_list, matched_atom_list, noise_matched_atom_list = [], [], []
    matched_bond_list, noise_matched_bond_list = [], []

    for i, smiles in enumerate(smiles_list):
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = smiles
        mol_list.append(mol)
        if smarts_list is not None:
            patt = Chem.MolFromSmarts(smarts_list[i])
            atoms_matched = mol.GetSubstructMatch(patt)
            bonds_matched = get_bonds(mol, atoms_matched)
            matched_atom_list.append(atoms_matched)
            matched_bond_list.append(bonds_matched)

        if noise_smarts_list is not None:
            patt = Chem.MolFromSmarts(noise_smarts_list[i])
            atoms_matched = mol.GetSubstructMatch(patt)
            bonds_matched = get_bonds(mol, atoms_matched)
            noise_matched_atom_list.append(atoms_matched)
            noise_matched_bond_list.append(bonds_matched)

        if smarts_list is None and highlight:
            atoms_matched = []
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
                    atoms_matched.append(atom.GetIdx())
            bonds_matched = get_bonds(mol, atoms_matched)
            matched_atom_list.append(atoms_matched)
            matched_bond_list.append(bonds_matched)

    all_matched_atom_list, all_matched_bond_list = [], []
    all_matched_atom_color, all_matched_bond_color = [], []

    for i in range(len(matched_atom_list)):
        if len(noise_matched_atom_list):
            all_matched_atom_list.append(matched_atom_list[i] + noise_matched_atom_list[i])
            all_matched_bond_list.append(matched_bond_list[i] + noise_matched_bond_list[i])
            atom2color = {a: color1 for a in matched_atom_list[i]}
            atom2color.update({a: color2 for a in noise_matched_atom_list[i]})
            bond2color = {b: color1 for b in matched_bond_list[i]}
            bond2color.update({b: color2 for b in noise_matched_bond_list[i]})
        else:
            all_matched_atom_list.append(matched_atom_list[i])
            all_matched_bond_list.append(matched_bond_list[i])
            atom2color = {a: color1 for a in matched_atom_list[i]}
            bond2color = {b: color1 for b in matched_bond_list[i]}

        all_matched_atom_color.append(atom2color)
        all_matched_bond_color.append(bond2color)

    if anchor_smi is not None:
        s = Chem.MolFromSmiles(anchor_smi)
        AllChem.Compute2DCoords(s)
        for m in mol_list:
            AllChem.GenerateDepictionMatching2DStructure(m, s, acceptFailure=True, forceRDKit=True, allowRGroups=True)

    if straighten:
        for mol in mol_list:
            Chem.rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(mol)
            rdDepictor.StraightenDepiction(mol)

    svg = Draw.MolsToGridImage(mol_list, subImgSize=size, useSVG=True, molsPerRow=2,
                               highlightAtomLists=all_matched_atom_list,
                               highlightAtomColors=all_matched_atom_color,
                               highlightBondLists=all_matched_bond_list,
                               highlightBondColors=all_matched_bond_color, maxMols=99999)
    # svg = SVG(svg)
    display(svg)
    if save_path:
        svg2png(bytestring=svg.data, write_to=save_path, dpi=300)


def draw_mol_by_reaction_center(smiles, atom_scores, bond_scores=None, size=(400, 400), color_name='Oranges'):
    cmap = plt.get_cmap(color_name)
    mol = Chem.MolFromSmiles(smiles)
    plot_atoms_colors, plot_bonds_colors = [], []
    plot_atoms_matched, plot_bonds_matched = [], []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom_score = atom_scores[int(atom.GetProp('molAtomMapNumber'))]
            plot_atoms_colors.append(cmap(atom_score))
            plot_atoms_matched.append(atom.GetIdx())

            if bond_scores is not None:
                for n_atom in atom.GetNeighbors():
                    bond_score = bond_scores[int(atom.GetProp('molAtomMapNumber')),
                                             int(n_atom.GetProp('molAtomMapNumber'))]
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), n_atom.GetIdx())
                    plot_bonds_colors.append(cmap(bond_score))
                    plot_bonds_matched.append(bond.GetIdx())

    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')

    atom2color = {plot_atoms_matched[i]: plot_atoms_colors[i] for i in range(len(plot_atoms_colors))}
    bond2color = {plot_bonds_matched[i]: plot_bonds_colors[i] for i in range(len(plot_bonds_colors))}

    svg = Draw.MolsToGridImage([mol], subImgSize=size, useSVG=True, molsPerRow=2,
                               highlightAtomLists=[plot_atoms_matched],
                               highlightAtomColors=[atom2color],
                               highlightBondLists=[plot_bonds_matched],
                               highlightBondColors=[bond2color])
    display(svg)


def draw_mol_with_highlight(smi, atom_scores, size=(250, 250), enable_bond=False, color_name='Oranges', save_path='', straighten=False):
    cmap = plt.get_cmap(color_name)
    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = smi
    plot_atoms_colors = []
    plot_atoms_matched = []
    plot_bonds_colors = []
    plot_bonds_matched = []

    for bond in mol.GetBonds():
        plot_bonds_matched.append(bond.GetIdx())
        plot_bonds_colors.append(cmap(0.0))

    for i, atom in enumerate(mol.GetAtoms()):
        plot_atoms_colors.append(cmap(atom_scores[i]))
        plot_atoms_matched.append(atom.GetIdx())

        if enable_bond and atom_scores[i] > 0:
            for n_atom in atom.GetNeighbors():
                if atom_scores[n_atom.GetIdx()] > 0:
                    bond_tmp = mol.GetBondBetweenAtoms(n_atom.GetIdx(), atom.GetIdx())
                    plot_bonds_colors[bond_tmp.GetIdx()] = cmap(0.5)

    atom2color = {plot_atoms_matched[i]: plot_atoms_colors[i] for i in range(len(plot_atoms_colors))}
    bond2color = {plot_bonds_matched[i]: plot_bonds_colors[i] for i in range(len(plot_bonds_colors))}

    if straighten:
        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)

    svg = Draw.MolsToGridImage([mol], subImgSize=size, useSVG=True, molsPerRow=2,
                               highlightAtomLists=[plot_atoms_matched],
                               highlightAtomColors=[atom2color],
                               highlightBondLists=[plot_bonds_matched],
                               highlightBondColors=[bond2color])

    display(svg)
    if save_path:
        svg2png(bytestring=svg.data, write_to=save_path, dpi=300)

def plot3d(Dx, ax, Y, prop_label="", rccounts=100, perplexity=5):
    print("projecting on 2D plane...")
    # mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    # X_2d = mds.fit_transform(squareform(Dx))

    tsne = TSNE(n_components=2, learning_rate='auto', init='random',
                metric='precomputed', random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(squareform(Dx))

    # get x,y,z for 3D plot
    x = X_2d[:, 0]
    y = X_2d[:, 1]
    z = Y

    # property max/min
    vmin = np.min(z)
    vmax = np.max(z)

    # get interpolation
    dx = 100j
    grid_x, grid_y = np.mgrid[x.min():x.max():dx, y.min():y.max():dx]
    grid_z = griddata(X_2d, z, (grid_x, grid_y), method='linear', rescale=True)

    # plot in 3D
    print("plotting...")
    masked_grid_z = np.ma.masked_invalid(grid_z)

    cmap = cm.get_cmap("coolwarm")
    colors = (masked_grid_z - vmin) / (vmax - vmin)
    facecolors = cmap(colors)

    ax.plot_surface(grid_x, grid_y, masked_grid_z, rcount=rccounts, ccount=rccounts,
                    color='b', alpha=0.75, linewidth=0, edgecolors='w', zorder=10,
                    facecolors=facecolors, vmin=vmin, vmax=vmax)

    ax.set_xlabel(r'$z_1$', labelpad=8)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel(r'$z_2$', labelpad=8)
    ax.set_ylim(y.min(), y.max())
    ax.set_zlabel('Z')
    ax.set_zlim(z.min(), z.max())

    ax.view_init(45, -60)

    ax.set_zlabel(prop_label, labelpad=12)
    ax.dist = 11

    return X_2d


def plot2d(X, z, ax=None, interpolation='linear', nlevels=8, scatter=False, density=True, cbar=False,
           cont_lw=0.75, cont_ls='solid', cont_colors='k', scatter_edgecolor='k', scatter_alpha=0.5,
           cmap='RdBu', vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Interpolate and generate heatmap:
    x = X[:, 0]
    y = X[:, 1]

    grid_x, grid_y = np.mgrid[x.min():x.max():1000j, y.min():y.max():1000j]
    grid_z = griddata(X, z, (grid_x, grid_y), method=interpolation, rescale=True)

    # g = ax.pcolormesh(grid_x, grid_y, np.ma.masked_invalid(grid_z), cmap='Greens_r', vmin=np.nanmin(grid_z), vmax=np.nanmax(grid_z))
    if vmin is None:
        vmin = np.nanmin(grid_z)
    if vmax is None:
        vmax = np.nanmax(grid_z)

    g = ax.contourf(grid_x, grid_y, np.ma.masked_invalid(grid_z), levels=nlevels, cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=1)

    contours = ax.contour(grid_x, grid_y, np.ma.masked_invalid(grid_z), nlevels, colors=cont_colors, linewidths=cont_lw,
                          linestyles=cont_ls)

    if scatter is True:
        ax.scatter(x, y, s=20, color='none', linewidth=0.5, edgecolor=scatter_edgecolor, alpha=scatter_alpha, zorder=10)

    if density is True:
        _reference_colors = [[1., 1., 1., 0.], [1., 1., 1., 0.9]]
        _cmap = LinearSegmentedColormap.from_list('white', _reference_colors)
        _cmap_r = LinearSegmentedColormap.from_list('white_r', _reference_colors[::-1])

        # add density as white shade
        kde = gaussian_kde(dataset=np.array([x, y]), bw_method=None)
        density = kde(np.c_[grid_x.flat, grid_y.flat].T).reshape(grid_x.shape)
        cset = ax.contourf(grid_x, grid_y, density, cmap=_cmap_r, vmin=np.nanmin(density), vmax=np.nanmax(density))

    ax.axis('off')

    if cbar is True:
        _ = fig.colorbar(g)

    return g