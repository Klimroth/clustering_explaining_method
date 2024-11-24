# basic configuration
INPUT_FILE_OBSERVABLE_FEATURES = "../data/biological_set_raw_data.xlsx"
INPUT_FILE_EXPLAINING_FEATURES = "../data/biological_set_explaining_features.xlsx"

OUTPUT_FOLDER_BASE = (
    '../results/naturverbundenheit/'
)

GROUP_NAME = "Individual"
OBSERVABLE_NAME = "observations (night)"
DATASET_NAME = "biological-dataset"
OBSERVABLE_PATTERN_NAME = "behavioral pattern"
OBSERVABLE_PATTERN_NAME_PLURAL = "behavioral patterns"

OBSERVABLE_FEATURE_NAMES = {
    "num_lhd": "# LHD phases",
    "num_lying": "# lying phases",
    "perc_lhd": "proportion LHD",
    "perc_lying": "proportion lying",
}

EXPLAINING_FEATURE_NAMES = {
    x: x
    for x in [
        "Age",
        "Sex",
        "Zoo",
        "Stable",
        "Genus_ID",
        "Family_ID",
        "Order_ID",
        "SH",
        "Weight",
        "Habitat",
    ]
}

# GROUP_NAME = "Country"
# OBSERVABLE_NAME = "indices"
# DATASET_NAME = "nature-dataset"
# OBSERVABLE_PATTERN_NAME = "response type"
# OBSERVABLE_PATTERN_NAME_PLURAL = "response types"
#
# OBSERVABLE_FEATURE_NAMES = {
#    "Difference": "discrimination ability",
#    "Env_Pol_1": "pollution",
#    "Climate_1": "climate change",
#    "Aliens_1": "invasive species",
#    "Landuse_1": "landuse",
#    "Exploit_1": "exploitation of natural resources"
# }
#
#
# EXPLAINING_FEATURE_NAMES = {
#    x:x for x in ["GBIS", "NOIS", "APMP", "FosCO2", "CO2Em", "ConvVegToCrop", "BHI", "AQ", "EH", "NFL", "UP", "HDI", "LPI", "EPI", "GII", "SDG", "NRI", "FA", "AL", "GI"]
# }


# determining the maximum number of observable patterns
GAP_STATISTIC_CLUSTER_RANGE = 10

# pattern calculation
# Either a number or 'auto'
# If 'auto' is chosen, the algorithm will take the minimum from the gap-statistic and an elbow-estimate 
NUMBER_OBSERVABLE_PATTERNS = 'auto'

# if files already exist, do not compute them again
USE_CACHED_DATASET = False
USE_CACHED_CLUSTERING = False

# distortion parameters
DISTORTION_MEAN = 0
DISTORTION_STD = 0.001

# imputation
NN_IMPUTATION_K = 10

# distance measure of fingerprints: jensenshannon, euclidean, correlation
DISTANCE_MEASURE_FINGERPRINT = "jensenshannon"
DISTANCE_MEASURE_EXPLAINABLE_FEATURES = "correlation"

# mode of infering the optimal set of explaining features exact/heuristic
INFERENCE_MODE_EXPLAINING_FEATURES = "exact"

# maximum parallel threads
MAX_NUM_THREADS = 12

# The spiderplots will be scaled before plotting
# One of 'minmax', 'none'
SPIDERPLOT_SCALING = 'none'

# If True, the fingerprints in the spiderplots will be rescaled
# So that the sum of all vectors will be 1.
ADJ_SCALE = False

# A value greater or equal 0 but smaller than 1.
# This parameter applies a penalty for using more explaining parameters.
# 0 means no penalty, whereas a higher number implies a higher penalty.
SPARSITY = 0.05