from pathlib import Path

RANDOM_STATE = 42
ROBUSTNESS_SEEDS = [42, 7, 21, 100, 2025]

ILPD_COLUMNS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Selector",
]

# Original: 1 = liver disease, 2 = no liver disease
# Converted: 1 = liver disease (positive class), 0 = no liver disease
TARGET_MAPPING = {1: 1, 2: 0}
CLASS_LABELS = ["No liver disease", "Liver disease"]

CATEGORICAL_FEATURES = ["Gender"]

DEMOGRAPHIC_FEATURES = ["Age", "Gender"]
BILIRUBIN_FEATURES = ["Total_Bilirubin", "Direct_Bilirubin"]
ENZYME_FEATURES = [
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
]
PROTEIN_FEATURES = [
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]

# Number of candidate models to carry forward from NB03 to NB04
N_CANDIDATES = 2


def get_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def get_project_paths(root: Path | None = None):
    if root is None:
        root = get_project_root()

    paths = {
        "root": root,
        "raw_data_path": root / "data" / "raw" / "ilpd.csv",
        "processed_data_path": root / "data" / "processed" / "ilpd_clean.csv",
        "tables_dir": root / "results" / "tables",
        "figures_dir": root / "results" / "figures",
        "models_dir": root / "results" / "models",
    }

    for key in ("tables_dir", "figures_dir", "models_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    return paths
