from biotite.structure import array, Atom, dihedral
from biotite.structure.io.pdb import PDBFile
from small_protein_gen.utils.data import _place_atom
import torch
from torch import Tensor
from typing import Dict


class ProteinStructure:
    def __init__(self, atom_pos: Dict[str, Tensor]) -> None:
        self.atom_pos = atom_pos

    def from_backbone(bb: Tensor) -> "ProteinStructure":
        """
        Args:
          bb: (A x N x 3) backbone 3D coordinates (A=3 for N, CA, C)
        """
        n_atoms, ca_atoms, c_atoms = bb.unbind(dim=0)
        atom_pos = {"N": n_atoms, "CA": ca_atoms, "C": c_atoms}
        return ProteinStructure(atom_pos)

    def _impute_oxygen(self) -> None:
        N_ATOMS = self.atom_pos["CA"].shape[0] - 1
        # TODO: why does negating bond angle (to become -120) lead to correct orientation?
        C_O_BOND_ANGLE = torch.tensor([-torch.pi * 120 / 180] * N_ATOMS)
        C_O_BOND_LENGTH = torch.tensor([1.23] * N_ATOMS)

        prev_N = self.atom_pos["N"][:-1]
        prev_CA = self.atom_pos["CA"][:-1]
        prev_C = self.atom_pos["C"][:-1]
        next_N = self.atom_pos["N"][1:]

        psi = torch.tensor(dihedral(prev_N, prev_CA, prev_C, next_N))

        O = _place_atom(
            prev_N,
            prev_CA,
            prev_C,
            torsion_angle=psi,
            bond_angle=C_O_BOND_ANGLE,
            bond_length=C_O_BOND_LENGTH,
        )
        self.atom_pos["O"] = O

    def to_pdb(self, file: str, impute_oxygen: bool = True) -> None:
        N_RES = self.atom_pos["CA"].shape[0]
        BB_ATOMS = ["N", "CA", "C", "O"]
        ATOM_TO_ELEMENT = {"N": "N", "CA": "C", "C": "C", "O": "O"}

        if impute_oxygen:
            self._impute_oxygen()

        out = []
        for i in range(N_RES):
            res_id = i + 1
            for atom in BB_ATOMS:
                if atom not in self.atom_pos or i >= len(self.atom_pos[atom]):
                    continue
                out.append(
                    Atom(
                        self.atom_pos[atom][i].numpy(),
                        chain_id="A",
                        res_id=res_id,
                        ins_code="",
                        res_name="ALA",
                        atom_name=atom,
                        element=ATOM_TO_ELEMENT[atom],
                    )
                )
        out = array(out)
        f = PDBFile()
        f.set_structure(out)
        f.write(file)
