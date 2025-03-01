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
from typing import Any, Callable, List, Tuple, TypeVar
import warnings

ProteinFeature = TypeVar("ProteinFeature")

N_ATOM = "N"
CA_ATOM = "CA"
C_ATOM = "C"
PDB_SUFFIX = ".pdb"

to_tensor = torch.from_numpy


def index_wrapper(
    i: int, callback: Callable[[Any], Any], *args: List[Any]
) -> Tuple[Any, int]:
    return callback(*args), i


def process_pdbs_in_directory(
    data_path: str, callback: Callable[[AtomArray], ProteinFeature], n_workers: int = 8
) -> List[ProteinFeature]:
    futures = []
    i = 0
    with ProcessPoolExecutor(n_workers) as executor:
        for file in os.scandir(data_path):
            if not file.is_file() or not file.name.endswith(PDB_SUFFIX):
                continue
            f_pdb = PDBFile.read(file)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Only interested in monomers
                atom_arr = f_pdb.get_structure(model=1)
            futures.append(executor.submit(index_wrapper, i, callback, atom_arr))
            i += 1
    out = [None] * i
    for future in as_completed(futures):
        if future.exception() is None:
            result, i = future.result()
            out[i] = result
        else:
            raise future.exception()
    return out


def get_backbone_angles(
    atom_arr: AtomArray, max_n_residues: int = 128
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      (N x 6) backbone angles matrix.
    """
    bb_atoms = atom_arr[filter_peptide_backbone(atom_arr)]
    N_RES = len(bb_atoms[bb_atoms.atom_name == CA_ATOM])

    bb_angles = torch.full((max_n_residues - 1, 6), torch.nan)
    mask = torch.zeros(max_n_residues - 1).long()
    mask[: N_RES - 1] = 1

    # Get torsion angles
    phi, psi, omega = map(to_tensor, dihedral_backbone(bb_atoms))
    torsion_angles = torch.stack([phi[1:], psi[:-1], omega[:-1]], dim=1)

    # Get bond angles (theta_1, theta_2, theta_3)
    n_atoms, ca_atoms, c_atoms = (
        bb_atoms[bb_atoms.atom_name == atom] for atom in (N_ATOM, CA_ATOM, C_ATOM)
    )
    bond_angles = torch.full((N_RES - 1, 3), torch.nan)
    bond_angles[:, 0] = to_tensor(angle(n_atoms[1:], ca_atoms[1:], c_atoms[1:]))
    bond_angles[:, 1] = to_tensor(angle(ca_atoms[:-1], c_atoms[:-1], n_atoms[1:]))
    bond_angles[:, 2] = to_tensor(angle(c_atoms[:-1], n_atoms[1:], ca_atoms[1:]))

    bb_angles[: N_RES - 1] = torch.cat([torsion_angles, bond_angles], dim=1)
    bb_angles[mask == 0] = 0

    return bb_angles, mask


get_backbone_angles_from_directory = partial(
    process_pdbs_in_directory, callback=get_backbone_angles
)


def _place_atom(
    atom1: Tensor,
    atom2: Tensor,
    atom3: Tensor,
    torsion_angle: Tensor,
    bond_angle: Tensor,
    bond_length: Tensor,
) -> Tensor:
    """Places next atom given previous three atoms, angles, and bond length
    Args:
      (B x 3) atom #1 coordinates
      (B x 3) atom #2 coordinates
      (B x 3) atom #3 coordinates
      (B) torsion angles
      (B) bond angles
      (B) bond lengths
    Returns:
      (B x 3) next atom coordinates
    """
    bond1 = atom2 - atom1
    bond2 = atom3 - atom2
    bond2 = bond2 / torch.norm(bond2, dim=1, keepdim=True)

    cross1 = torch.cross(bond1, bond2, dim=-1)
    cross1 = cross1 / torch.norm(cross1, dim=-1, keepdim=True)
    cross2 = torch.cross(cross1, bond2, dim=-1)

    m = torch.stack([bond2, cross2, cross1], dim=-1)  # B x 3 x 3

    d = bond_length.unsqueeze(-1) * torch.stack(
        [
            -torch.cos(bond_angle),
            torch.cos(torsion_angle) * torch.sin(bond_angle),
            torch.sin(torsion_angle) * torch.sin(bond_angle),
        ],
        dim=-1,
    )  # B x 3

    next_atom = torch.einsum("bij,bj->bi", m, d) + atom3

    return next_atom


def angles_to_backbone(angle: Tensor, mask: Tensor) -> Tensor:
    """Uses the NeRF algorithm to rebuild protein backbone
    Args:
      (B x N x 6) batched backbone angles tensor
      (B x N) batched backbone angles mask
    Returns:
      (B x A x N x 3) backbone 3D coordinates (A=3 for N, CA, C)
    """
    N_BATCH, N_RES = mask.shape

    # Constants as per FoldingDiff
    BOND_LENGTHS = torch.tensor([1.46, 1.54, 1.34])  # N-CA, CA-C, C-N

    ## Initial three atoms (from 1CRN)
    N_INIT = torch.tensor([17.047, 14.099, 3.625])
    CA_INIT = torch.tensor([16.967, 12.784, 4.338])
    C_INIT = torch.tensor([15.685, 12.755, 5.133])

    out = torch.zeros((N_BATCH, 3, N_RES + 1, 3))
    out[:, 0, 0] = N_INIT
    out[:, 1, 0] = CA_INIT
    out[:, 2, 0] = C_INIT

    for i in range(N_RES):
        incomplete = torch.argwhere(mask[:, i])
        if incomplete.numel() == 0:
            break
        incomplete = incomplete[:, 0]
        prev_N, prev_CA, prev_C = out[incomplete, :, i].unbind(dim=1)
        phi, psi, omega, theta1, theta2, theta3 = angle[incomplete, i].unbind(dim=-1)

        new_N = _place_atom(prev_N, prev_CA, prev_C, psi, theta2, BOND_LENGTHS[2])
        new_CA = _place_atom(prev_CA, prev_C, new_N, omega, theta3, BOND_LENGTHS[0])
        new_C = _place_atom(prev_C, new_N, new_CA, phi, theta1, BOND_LENGTHS[1])

        out[incomplete, 0, i + 1] = new_N
        out[incomplete, 1, i + 1] = new_CA
        out[incomplete, 2, i + 1] = new_C

    return out
