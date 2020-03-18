BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7
ONE_HALF = 0.5
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'

LOSS = 'loss'
ACCURACY = 'accuracy'
OPTIMIZER_OP = 'optimizer_op'
F1_SCORE = 'f1_score'
GLOBAL_STEP = 'global_step'

# Data format constants
SAMPLE_ID = 'sample_id'
DATA_FIELD_FORMAT = '{0}-{1}'
INPUTS = 'inputs'
OUTPUT = 'output'
DATA_FIELDS = [INPUTS, OUTPUT]
INDEX_FILE = 'index.pkl.gz'

# Data folder names
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

# File Names
NAME_FMT = '{0}-{1}-{2}_model_best'
HYPERS_PATH = 'model-hyper-params-{0}.pkl.gz'
METADATA_PATH = 'model-metadata-{0}.pkl.gz'
MODEL_PATH = 'model-{0}.pkl.gz'
TRAIN_LOG_PATH = 'model-train-log-{0}.pkl.gz'
TEST_LOG_PATH = 'model-test-log-{0}.jsonl.gz'
OPTIMIZED_TEST_LOG_PATH = 'model-optimized-test-log-{0}.jsonl.gz'
MODEL_NAME_REGEX = r'^model-([^\.]+)(\.pkl\.gz)?$'
