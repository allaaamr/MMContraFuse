# preprocess_tcga.py
import argparse
import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------
def step(msg: str) -> None:
    print(f"\n[STEP] {msg}")

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def transform_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drops 'Entrez_Gene_Id' if present
    - Deduplicates by 'Hugo_Symbol'
    - Melts to long format with PatientID
    - Pivots back to PatientID x Hugo_Symbol wide matrix
    """
    if "Entrez_Gene_Id" in input_df.columns:
        input_df = input_df.drop(columns=["Entrez_Gene_Id"])

    if "Hugo_Symbol" not in input_df.columns:
        raise ValueError("Expected 'Hugo_Symbol' column in matrix file.")

    # Drop duplicate gene rows
    input_df = input_df.drop_duplicates(subset=["Hugo_Symbol"])

    # Melt to long form: columns except Hugo_Symbol become PatientIDs
    melted_df = input_df.melt(
        id_vars=["Hugo_Symbol"], var_name="PatientID", value_name="value"
    )

    # Pivot: PatientID rows, genes as columns
    transformed_df = melted_df.pivot(index="PatientID", columns="Hugo_Symbol", values="value")
    transformed_df.reset_index(inplace=True)
    transformed_df.columns.name = None  # remove pivot axis name
    return transformed_df


def non_intersecting_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[int, set]:
    """
    Returns (#non_intersecting, set_of_non_intersecting) for columns (includes PatientID).
    """
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)
    non_intersecting = (columns_df1 - columns_df2) | (columns_df2 - columns_df1)
    return len(non_intersecting), non_intersecting


def filter_genes(
    input_df: pd.DataFrame, allowed_genes: Iterable[str], extension: str
) -> pd.DataFrame:
    """
    Keep only columns that are in allowed_genes + 'PatientID', then suffix gene columns.
    """
    cols = list(input_df.columns)
    keep = ["PatientID"] + [g for g in cols if g in allowed_genes]
    out = input_df[keep].copy()
    # rename gene columns with suffix, keep PatientID unchanged
    rename_map = {c: f"{c}{extension}" for c in out.columns if c != "PatientID"}
    out = out.rename(columns=rename_map)
    return out


def read_tcga_matrix(path: str, delimiter: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=delimiter, comment="#", low_memory=False)
    return df


def load_signatures(csv_path: str) -> np.ndarray:
    """
    Load signatures CSV (columns are signature names, cells are gene symbols).
    Returns a 1D unique array of all gene symbols from all signature columns.
    """
    sig = pd.read_csv(csv_path, low_memory=False)
    genes: List[str] = []
    for col in sig.columns:
        vals = sig[col].dropna().astype(str).unique().tolist()
        genes.extend(vals)
    uniq = np.unique(genes)
    return uniq


def fix_gene_names(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename problematic gene symbols (e.g., MARC1->MTARC1) in columns (not PatientID).
    Only renames if a column exists.
    """
    rename_map = {k: v for k, v in mapping.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def filter_mut_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns certain columns in mutation matrix, and only keeping entries of missense mutations
    """
    columns_to_keep = ["Hugo_Symbol", "Consequence", "Variant_Classification", "Variant_Type", "Tumor_Sample_Barcode", "Matched_Norm_Sample_Barcode"]
    out = df[columns_to_keep].copy()
    out = out[out['Variant_Classification'] == 'Missense_Mutation' ]
    out.rename(columns={'Tumor_Sample_Barcode': 'case_id'}, inplace=True)
    # Remove the last 3 characters
    out['case_id'] = out['case_id'].str.slice(0, -3)

    return out

def transform_mut_matrix(df: pd.DataFrame, genes_to_keep) -> pd.DataFrame:
    patients =  df['case_id'].unique()
    mutation_matrix = pd.DataFrame(0, index=pd.Index(patients, name='case_id'), columns=genes_to_keep)
    for patient in patients:
        # Get the mutated genes for the current patient
        mutated_genes = df.loc[df['case_id'] == patient, 'Hugo_Symbol']
        # Set the corresponding entries in the mutation matrix to 1
        mutation_matrix.loc[patient, mutated_genes.unique()] = 1 
        # filter insignificant genes
        mutation_matrix = mutation_matrix[[col for col in mutation_matrix.columns if col in genes_to_keep]]

    return mutation_matrix

# -------------------------
# Main pipeline
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess TCGA GBM/LGG matrices (CNA/RNA) with signatures filter.")
    parser.add_argument("--gbm_dir", required=True, help="Directory containing GBM files")
    parser.add_argument("--lgg_dir", required=True, help="Directory containing LGG files")
    parser.add_argument("--signatures_csv", default="data/signatures.csv", help="CSV of signature genes")
    parser.add_argument("--outdir", default="data/processed_tabular_data", help="Output directory")
    parser.add_argument(
        "--modalities", nargs="+", required=True, choices=["clinical", "cna", "rna", "mut"],
        help="Which modalities to process"
    )
    parser.add_argument("--strict", action="store_true", help="Assert zero non-intersecting genes after rename")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # File names used
    CNA_FN = "data_cna.txt"
    RNA_FN = "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
    MUT_FN = "data_mutations.txt"  
    CLNC_FN = "data_clinical_patient.txt"

    # Optional gene symbol fixes 
    gene_renames = {
        "MARC1": "MTARC1",
        "MARC2": "MTARC2",
    }
    REQUIRED_COLS = [
    "PATIENT_ID", "SUBTYPE", "AGE", "SEX", "OS_STATUS", "OS_MONTHS", "RACE"
]

    RENAME_MAP = {
        "PATIENT_ID": "case_id",
        "SUBTYPE": "type",
        "AGE": "age",
        "SEX": "gender",
        "OS_MONTHS": "survival",
        "OS_STATUS": "censorship",
        "RACE": "race",
    }
    # -------------------------
    # Process clinical
    # -------------------------
    if "clinical" in args.modalities:
        step("Loading Clinical files from GBM and LGG")
        gbm_clin_path = os.path.join(args.gbm_dir, CLNC_FN)
        lgg_clin_path = os.path.join(args.lgg_dir, CLNC_FN)
        info(f"GBM clinical path: {gbm_clin_path}")
        info(f"LGG clinical path: {lgg_clin_path}")

        df_gbm = read_tcga_matrix(gbm_clin_path, delimiter="\t")
        df_lgg = read_tcga_matrix(lgg_clin_path, delimiter="\t")
        ok(f"Loaded CNA (GBM: {df_gbm.shape}, LGG: {df_lgg.shape})")

        step("Filtering to required columns ")
        print(df_gbm.columns)
        df_gbm = df_gbm[REQUIRED_COLS].copy()
        df_lgg = df_lgg[REQUIRED_COLS].copy()
        info(f"After filter shape: GBM {df_gbm.shape}, LGG {df_lgg.shape}")

        step("Renaming columns")
        df_gbm = df_gbm.rename(columns=RENAME_MAP)
        df_gbm["type"] = 1

        df_gbm["gender"] = np.where(df_gbm["gender"].eq("Female"), 1, 0)
        df_gbm["censorship"] = np.where(df_gbm["censorship"].eq("0:LIVING"), 0, 1)
        df_gbm["race"] = np.where(df_gbm["race"].eq("White"), 0, 1)

        df_lgg = df_lgg.rename(columns=RENAME_MAP)
        df_lgg["type"] = 0
        df_lgg["gender"] = np.where(df_lgg["gender"].eq("Female"), 1, 0)
        df_lgg["censorship"] = np.where(df_lgg["censorship"].eq("0:LIVING"), 0, 1)
        df_lgg["race"] = np.where(df_lgg["race"].eq("White"), 0, 1)


        step("Concatenating GBM + LGG clinical tables")
        df_clinical = pd.concat([df_lgg, df_gbm], ignore_index=True)
        info(f"Merged clinical shape: {df_clinical.shape}")

        step("Saving output CSV")
        df_clinical.to_csv("data/processed_tabular_data/clinical.csv", index=False)
        ok(f"Saved: data/processed_tabular_data/clinical.csv")


    # -------------------------
    # Load signatures genes
    # -------------------------
    step("Loading signatures")
    sig_genes = load_signatures(args.signatures_csv)
    info(f"Signatures loaded: {len(sig_genes)} unique genes")

    # -------------------------
    # Process CNA
    # -------------------------
    if "cna" in args.modalities:
        step("Loading raw CNA matrices")
        gbm_cna_path = os.path.join(args.gbm_dir, CNA_FN)
        lgg_cna_path = os.path.join(args.lgg_dir, CNA_FN)
        info(f"GBM CNA: {gbm_cna_path}")
        info(f"LGG CNA: {lgg_cna_path}")

        df_gbm_cna = read_tcga_matrix(gbm_cna_path, delimiter="\t")
        df_lgg_cna = read_tcga_matrix(lgg_cna_path, delimiter="\t")
        ok(f"Loaded CNA (GBM: {df_gbm_cna.shape}, LGG: {df_lgg_cna.shape})")

        step("Transforming CNA to PatientID x Gene wide format")
        gbm_cna_wide = transform_df(df_gbm_cna)
        lgg_cna_wide = transform_df(df_lgg_cna)
        ok(f"CNA transformed (GBM: {gbm_cna_wide.shape}, LGG: {lgg_cna_wide.shape})")

        step("Fixing gene symbol discrepancies for CNA (if any)")
        lgg_cna_wide = fix_gene_names(lgg_cna_wide, gene_renames)
        n_nonint, nonint = non_intersecting_columns(lgg_cna_wide.drop(columns=["PatientID"], errors="ignore"),
                                                    gbm_cna_wide.drop(columns=["PatientID"], errors="ignore"))
        info(f"CNA non-intersecting gene columns between LGG and GBM: {n_nonint}")
        if n_nonint > 0:
            warn(f"Non-intersecting CNA genes (sample): {list(sorted(nonint))[:10]}")
            if args.strict:
                raise AssertionError("CNA: Non-intersecting genes detected in strict mode.")

        step("Concatenating LGG + GBM CNA by rows")
        df_cna_all = pd.concat([lgg_cna_wide, gbm_cna_wide], axis=0, ignore_index=True)
        ok(f"CNA concatenated shape: {df_cna_all.shape}")

        step("Filtering CNA by signature genes and suffixing with _cna")
        intersect_count = len(set(df_cna_all.columns).intersection(set(sig_genes)))
        info(f"CNA: #genes intersecting with signatures BEFORE filter: {intersect_count}")
        df_cna = filter_genes(df_cna_all, sig_genes, "_cna")
        ok(f"CNA filtered shape: {df_cna.shape}")

        step("Saving CNA outputs")
        df_cna_all.to_csv(os.path.join(args.outdir, "combined_CNA_all.csv"), index=False)
        df_cna.to_csv(os.path.join(args.outdir, "combined_CNA_filtered.csv"), index=False)
        ok("CNA saved: combined_CNA_all.csv, combined_CNA_filtered.csv")

    # -------------------------
    # Process RNA
    # -------------------------
    if "rna" in args.modalities:
        step("Loading raw RNA matrices")
        gbm_rna_path = os.path.join(args.gbm_dir, RNA_FN)
        lgg_rna_path = os.path.join(args.lgg_dir, RNA_FN)
        info(f"GBM RNA: {gbm_rna_path}")
        info(f"LGG RNA: {lgg_rna_path}")

        df_gbm_rna = read_tcga_matrix(gbm_rna_path, delimiter="\t")
        df_lgg_rna = read_tcga_matrix(lgg_rna_path, delimiter="\t")
        ok(f"Loaded RNA (GBM: {df_gbm_rna.shape}, LGG: {df_lgg_rna.shape})")

        step("Transforming RNA to PatientID x Gene wide format")
        gbm_rna_wide = transform_df(df_gbm_rna)
        lgg_rna_wide = transform_df(df_lgg_rna)
        ok(f"RNA transformed (GBM: {gbm_rna_wide.shape}, LGG: {lgg_rna_wide.shape})")

        step("Fixing gene symbol discrepancies for RNA (if any)")
        lgg_rna_wide = fix_gene_names(lgg_rna_wide, gene_renames)
        n_nonint, nonint = non_intersecting_columns(lgg_rna_wide.drop(columns=["PatientID"], errors="ignore"),
                                                    gbm_rna_wide.drop(columns=["PatientID"], errors="ignore"))
        info(f"RNA non-intersecting gene columns between LGG and GBM: {n_nonint}")
        if n_nonint > 0:
            warn(f"Non-intersecting RNA genes (sample): {list(sorted(nonint))[:10]}")
            if args.strict:
                raise AssertionError("RNA: Non-intersecting genes detected in strict mode.")

        step("Concatenating LGG + GBM RNA by rows")
        df_rna_all = pd.concat([lgg_rna_wide, gbm_rna_wide], axis=0, ignore_index=True)
        ok(f"RNA concatenated shape: {df_rna_all.shape}")

        step("Filtering RNA by signature genes and suffixing with _rna")
        intersect_count = len(set(df_rna_all.columns).intersection(set(sig_genes)))
        info(f"RNA: #genes intersecting with signatures BEFORE filter: {intersect_count}")
        df_rna = filter_genes(df_rna_all, sig_genes, "_rna")
        ok(f"RNA filtered shape: {df_rna.shape}")

        step("Saving RNA outputs")
        df_rna_all.to_csv(os.path.join(args.outdir, "combined_RNA_all.csv"), index=False)
        df_rna.to_csv(os.path.join(args.outdir, "combined_RNA_filtered.csv"), index=False)
        ok("RNA saved: combined_RNA_all.csv, combined_RNA_filtered.csv")

    # -------------------------
    # Process Mutations
    # -------------------------
    if "mut" in args.modalities:
        step("Loading raw MUT matrices")
        gbm_mut_path = os.path.join(args.gbm_dir, MUT_FN)
        lgg_mut_path = os.path.join(args.lgg_dir, MUT_FN)
        info(f"GBM CNA: {gbm_mut_path}")
        info(f"LGG CNA: {lgg_mut_path}")

        df_gbm_mut = read_tcga_matrix(gbm_mut_path, delimiter="\t")
        df_lgg_mut = read_tcga_matrix(lgg_mut_path, delimiter="\t")
        ok(f"Loaded MUT (GBM: {df_gbm_mut.shape}, LGG: {df_lgg_mut.shape})")

        step("Dropping any missing entries")
        df_gbm_mut = df_gbm_mut.dropna(axis=1)
        df_lgg_mut = df_lgg_mut.dropna(axis=1)
        ok(f"Dropped Missing  (GBM: {df_gbm_mut.shape}, LGG: {df_lgg_mut.shape})")


        step("Filter Columns and Mutations")
        df_gbm_mut = filter_mut_columns(df_gbm_mut)
        df_lgg_mut = filter_mut_columns(df_lgg_mut)
        ok(f"(GBM: {df_gbm_mut.shape}, LGG: {df_lgg_mut.shape})")

        step("GBM Dataframe Creation")
        df_gbm = transform_mut_matrix(df_gbm_mut, sig_genes)
        ok(f"GBM MUT dataframe created  {df_gbm.shape}")

        step("LGG Dataframe Creation")
        df_lgg = transform_mut_matrix(df_lgg_mut, sig_genes)
        ok(f"GBM MUT dataframe created  {df_lgg.shape}")

        step("Combining LGG + GBM ")
        df_mut_all = pd.concat([df_lgg, df_gbm], axis=0, ignore_index=True)
        ok(f"MUT concatenated shape: {df_mut_all.shape}")

        step("Saving MUT outputs")
        df_mut_all.to_csv(os.path.join(args.outdir, "combined_MUT_filtered.csv"), index=False)
        ok("MUT saved: combined_MUT_filtered.csv")


    # -------------------------
    # Done
    # -------------------------
    elapsed = time.time() - t0
    ok(f"All done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
