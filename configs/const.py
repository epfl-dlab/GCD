import os

curr_file_dir = os.path.dirname(os.path.abspath(__file__))


CONFIG_DIR = curr_file_dir

HYDRACFG_DIR = os.path.join(CONFIG_DIR, "hydra_conf")

MODELS_DIR = os.environ["HF_MODELS_DIR"]

ASSETS_DIR = os.path.join(os.path.dirname(CONFIG_DIR), "assets")

PGF_DIR = os.path.join(ASSETS_DIR, "pgf")

DATA_DIR = os.path.join(os.path.dirname(CONFIG_DIR), "data")
