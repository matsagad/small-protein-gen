import argparse
from biotite.structure.io.pdb import PDBFile
import multiprocessing as mp
import os
import warnings

PDB_SUFFIX = ".pdb"
CA_ATOM = "CA"


def link_if_filtered(file: str, f_out: str, min_len: int, max_len: int) -> None:
    f_pdb = PDBFile.read(file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atom_arr = f_pdb.get_structure(model=1)
    n_residues = atom_arr[atom_arr.atom_name == CA_ATOM].shape[0]
    if min_len <= n_residues <= max_len:
        print(file, n_residues)
        os.symlink(file, os.path.join(f_out, os.path.basename(file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", required=True)
    parser.add_argument("-o", "--out_folder", required=True)
    parser.add_argument("--min_length", default=40)
    parser.add_argument("--max_length", default=100)
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
