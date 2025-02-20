from biotite.structure import (
    angle,
    AtomArray,
    dihedral_backbone,
    filter_peptide_backbone,
)
from biotite.structure.io.pdb import PDBFile
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
import os
import torch
from torch import Tensor
from typing import Callable, List, TypeVar
import warnings

ProteinFeature = TypeVar("ProteinFeature")

N_ATOM = "N"
CA_ATOM = "CA"
C_ATOM = "C"
PDB_SUFFIX = ".pdb"

to_tensor = torch.from_numpy


def process_pdbs_in_directory(
    data_path: str, callback: Callable[[AtomArray], ProteinFeature], n_workers: int = 8
) -> List[ProteinFeature]:
    futures = []
    with ProcessPoolExecutor(n_workers) as executor:
        for file in os.scandir(data_path):
            if not file.is_file() or not file.name.endswith(PDB_SUFFIX):
                continue
            f_pdb = PDBFile.read(file)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Only interested in monomers
                atom_arr = f_pdb.get_structure(model=1)
            futures.append(executor.submit(callback, atom_arr))
    out = []
    for future in as_completed(futures):
        out.append(future.result())
    return out


def get_backbone_angles(atom_arr: AtomArray, max_n_residues: int = 128) -> Tensor:
    """
    Returns:
      (N x 6) backbone angles matrix.
    """
    bb_atoms = atom_arr[filter_peptide_backbone(atom_arr)]
    N_RES = len(bb_atoms[bb_atoms.atom_name == CA_ATOM])

    bb_angles = torch.zeros(max_n_residues, 6)
    mask = torch.zeros(max_n_residues).long()
    mask[:N_RES] = 1

    # Get torsion angles (phi, psi, omega)
    torsion_angles = torch.stack(
        list(map(to_tensor, dihedral_backbone(bb_atoms))), dim=1
    )

    # Get bond angles (theta_1, theta_2, theta_3)
    n_atoms, ca_atoms, c_atoms = (
        bb_atoms[bb_atoms.atom_name == atom] for atom in (N_ATOM, CA_ATOM, C_ATOM)
    )
    bond_angles = torch.full((N_RES, 3), 0)
    bond_angles[:, 0] = to_tensor(angle(n_atoms, ca_atoms, c_atoms))
    bond_angles[:-1, 1] = to_tensor(angle(ca_atoms[:-1], c_atoms[1:], n_atoms[1:]))
    bond_angles[:-1, 2] = to_tensor(angle(c_atoms[1:], n_atoms[1:], ca_atoms[1:]))

    bb_angles[:N_RES] = torch.cat([torsion_angles, bond_angles], dim=1)
    bb_angles[bb_angles.isnan()] = 0

    return bb_angles, mask


get_backbone_angles_from_directory = partial(
    process_pdbs_in_directory, callback=get_backbone_angles
)
