from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'
TRAIN_FEATURES_DIR = BASE_DIR / 'train_features'
JOBS_DIR = BASE_DIR / 'jobs'
SIDECHAINNET_DATA_DIR = BASE_DIR / 'sidechainnet_data'
SIDECHAINNET_TEST_DIR = SIDECHAINNET_DATA_DIR / 'test'
TEMP_DIR = BASE_DIR / 'temp'

