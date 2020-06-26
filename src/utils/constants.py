BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7
ONE_HALF = 0.5
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'

# Tensorflow names
LOSS = 'loss'
ACCURACY = 'accuracy'
OPTIMIZER_OP = 'optimizer_op'
F1_SCORE = 'f1_score'
LOGITS = 'logits'
GLOBAL_STEP = 'global_step'
PREDICTION = 'prediction'
DROPOUT_KEEP_RATE = 'dropout-keep-rate'
NODE_REGEX_FORMAT = '.*{0}.*'

# Data format constants
SAMPLE_ID = 'sample_id'
DATA_FIELD_FORMAT = '{0}-{1}'
INPUTS = 'inputs'
OUTPUT = 'output'
TIMESTAMP = 'timestamp'
DATA_FIELDS = [INPUTS, OUTPUT]
INDEX_FILE = 'index.pkl.gz'

# Metadata Constants
INPUT_SCALER = 'input_scaler'
OUTPUT_SCALER = 'output_scaler'
INPUT_SHAPE = 'input_shape'
NUM_OUTPUT_FEATURES = 'num_output_features'
SEQ_LENGTH = 'seq_length'
INPUT_NOISE = 'input_noise'
NUM_CLASSES = 'num_classes'
LABEL_MAP = 'label_map'
REV_LABEL_MAP = 'rev_label_map'

# Model type names
MODEL = 'model'
SCHEDULED_MODEL = 'scheduled_model'
SCHEDULED_GENETIC = 'scheduled_genetic'
SCHEDULED_OPTIMIZED = 'scheduled_optimized'
OPTIMIZED_RESULTS = 'optimized_results'

# Compression Hashing Seeds (kept short for efficiency purposes)
TRANSFORM_SEED = 'tr'
EMBEDDING_SEED = 'em'
AGGREGATE_SEED = 'ag'
OUTPUT_SEED = 'ou'
UPDATE_SEED = 'up'
RESET_SEED = 'rs'
CANDIDATE_SEED = 'cd'
FUSION_SEED = 'fs'
BACKWARD_SEED = 'bw'

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
FINAL_VALID_LOG_PATH = 'model-final-valid-log-{0}.jsonl.gz'
FINAL_TRAIN_LOG_PATH = 'model-final-train-log-{0}.jsonl.gz'
TEST_LOG_PATH = 'model-test-log-{0}.jsonl.gz'
OPTIMIZED_TEST_LOG_PATH = 'model-optimized-{0}-{1}-{2}-test-log-{3}.jsonl.gz'
GRAPH_PATH = 'model-graph-{0}.pbtxt'
MODEL_NAME_REGEX = r'^model-([^\.]+)(\.pkl\.gz)?$'
