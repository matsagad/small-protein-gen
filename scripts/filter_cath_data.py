import argparse
from biotite.structure.io.pdb import PDBFile
from biotite.structure.geometry import (
    dihedral_backbone,
    filter_peptide_backbone,
    BadStructureError,
)
import multiprocessing as mp
import os
import warnings

N_ATOM = "N"
CA_ATOM = "CA"
C_ATOM = "C"
PDB_SUFFIX = ".pdb"


def link_if_filtered(file: str, f_out: str, min_len: int, max_len: int) -> None:
    f_pdb = PDBFile.read(file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atom_arr = f_pdb.get_structure(model=1)
    n_residues = atom_arr[atom_arr.atom_name == CA_ATOM].shape[0]
    if min_len <= n_residues <= max_len:
        try:
            dihedral_backbone(atom_arr)
        except BadStructureError:
            # Filter out if backbone is invalid
            return
        bb_atoms = atom_arr[filter_peptide_backbone(atom_arr)]
        n_atoms, ca_atoms, c_atoms = (
            bb_atoms[bb_atoms.atom_name == atom] for atom in (N_ATOM, CA_ATOM, C_ATOM)
        )
        if not (n_atoms.shape == ca_atoms.shape == c_atoms.shape):
            return
        res_ids = ca_atoms.get_annotation("res_id")
        if not (res_ids[:-1] + 1 == res_ids[1:]).all():
            # Filter out if sequence id is not monotonic
            return

        os.symlink(os.path.realpath(file), os.path.join(f_out, os.path.basename(file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", required=True)
    parser.add_argument("-o", "--out_folder", required=True)
    parser.add_argument("--min_length", default=40)
    parser.add_argument("--max_length", default=128)
    parser.add_argument("--n_workers", default=8)
    args = parser.parse_args()

    f_data = args.data_folder
    f_out = args.out_folder
    min_len = args.min_length
    max_len = args.max_length
    n_workers = args.n_workers

    if not os.path.exists(f_out):
        os.makedirs(f_out)

    pool = mp.Pool(n_workers)
    for file in os.scandir(f_data):
        if not file.is_file() or not file.name.endswith(PDB_SUFFIX):
            continue
        pool.apply_async(link_if_filtered, args=(file.path, f_out, min_len, max_len))
    pool.close()
    pool.join()
