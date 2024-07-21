# basic configuration
INPUT_FILE_OBSERVABLE_FEATURES = ""
INPUT_FILE_EXPLAINING_FEATURES = ""

OUTPUT_FOLDER_BASE = ""

GROUP_NAME = "Individual"
DATASET_NAME = "biological-dataset"

OBSERVABLE_FEATURE_NAMES = {
    "num_lhd": "# LHD phases",
    "num_lying": "# lying phases",
    "perc_lhd": "proportion LHD",
    "perc_lying": "proportion lying",
}

EXPLAINING_FEATURE_NAMES = {}

# determining the number of observable patterns
GAP_STATISTIC_CLUSTER_RANGE = 10

# pattern calculation
NUMBER_OBSERVABLE_PATTERNS = 5

# if files already exist, do not compute them again
USE_CACHED_DATASET = True
USE_CACHED_CLUSTERING = True

# distortion parameters
DISTORTION_MEAN = 0
DISTORTION_STD = 0.01

# imputation
NN_IMPUTATION_K = 10
