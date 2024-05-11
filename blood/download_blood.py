import os
import cellxgene_census

VERSION = "2023-07-25"

with cellxgene_census.open_soma(census_version=VERSION) as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        obs_value_filter="suspension_type != 'na' and disease == 'normal' and tissue_general in ['blood']",
    )
    print("adata object created")

    query_adata_path = os.path.join("dataset", f"blood.h5ad")
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    adata.write_h5ad(query_adata_path)
    print("Partition downloaded")
