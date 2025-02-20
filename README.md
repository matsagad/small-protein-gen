# small-protein-gen

A collection of generative models on small proteins (<=128) trained on my macbook pro.

## Models

Models are labelled according to their reference's name, but implementations (and hence performance) probably differ with the original.

| Name | Description | Reference |
|:----:|:------------|:---------:|
| `foldingdiff`| Diffuse backbone torsion and bond angles in $\mathbb{T}^{6\times (N - 1)}$. | [Wu et al. (2024)](https://www.nature.com/articles/s41467-024-45051-2) |


## Data

Dataset used is a non-redundant version of the CATH dataset with max 40% sequence similarity. Chains are filtered to be within 40-128 residues in length (no random crops). This results in a total of 15,696 monomers.

## Feedback

Feedback is welcome! Raise an issue or drop a PR.