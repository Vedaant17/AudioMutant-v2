import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset and index paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
INDEX_DIR = os.path.join(BASE_DIR, "indices")

# Versioning
INDEX_VERSION = "v1"
INDEX_PATH = os.path.join(INDEX_DIR, f"index_{INDEX_VERSION}.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, f"metadata_{INDEX_VERSION}.pkl")

# FAISS parameters
N_LIST = 100      # Number of clusters for IVF
N_PROBE = 10      # Number of clusters searched during query
TOP_K_DEFAULT = 5

# Hybrid retrieval weights
EMBEDDING_WEIGHT = 0.7
FEATURE_WEIGHT = 0.3

# Ensure directories exist
os.makedirs(INDEX_DIR, exist_ok=True)