# basic configuration
INPUT_FILE_OBSERVABLE_FEATURES = "../data/biological_set_raw_data.xlsx"
INPUT_FILE_EXPLAINING_FEATURES = "../data/biological_set_explaining_features.xlsx"
#INPUT_FILE_OBSERVABLE_FEATURES = "../data/naturverbundenheit_daten_modified.xlsx"
#INPUT_FILE_EXPLAINING_FEATURES = "../data/naturverbundenheit_indices.xlsx"

OUTPUT_FOLDER_BASE = (
    '../results/zoo_1/'
)

GROUP_NAME = "Individual"
OBSERVABLE_NAME = "observations (night)"
DATASET_NAME = "biological-dataset"
OBSERVABLE_PATTERN_NAME = "behavioral pattern"
OBSERVABLE_PATTERN_NAME_PLURAL = "behavioral patterns"

OBSERVABLE_FEATURE_NAMES = {
    #"num_lhd": "# LHD phases",
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
        "SH",
        "Weight",
        "Habitat",
    ]
}

#GROUP_NAME = "Country"
#OBSERVABLE_NAME = "indices"
#DATASET_NAME = "nature-dataset"
#OBSERVABLE_PATTERN_NAME = "response type"
#OBSERVABLE_PATTERN_NAME_PLURAL = "response types"

'''OBSERVABLE_FEATURE_NAMES = {
    "Difference": "discrimination ability",
    "Env_Pol_1": "pollution",
    "Climate_1": "climate change",
    "Aliens_1": "invasive species",
    "Landuse_1": "landuse",
    "Exploit_1": "exploitation of natural resources"
}'''

#OBSERVABLE_FEATURE_NAMES = {
#    "Env_Pol_1": "pollution",
#    "Climate_1": "climate change",
#    "Aliens_1": "invasive species",
#    "Landuse_1": "landuse",
#    "Exploit_1": "exploitation of natural resources"
#}

#EXPLAINING_FEATURE_NAMES = {
#   x:x for x in ["NOIS", "APMP", "FosCO2", "BHI", "AQ", "EH", "NFL", "UP", "HDI", "LPI", "EPI", "SDG", "NRI", "FA", "AL", "GI"]
#}


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
INFERENCE_MODE_EXPLAINING_FEATURES = "exact" #"heuristic"

# if mode 'heurisitc' is chosen, the algorithm will compute the exact values
# for feature combinations up do a combinatory depth of N; from there, it will
# perform a greedy search.
HEURISTIC_N = 3

# maximum parallel threads
MAX_NUM_THREADS = 12

# A value greater or equal 0 but smaller than 1.
# This parameter applies a penalty for using more explaining parameters.
# 0 means no penalty, whereas a higher number implies a higher penalty.
SPARSITY = 0.00

# Higher-order permutation-based feature importance K.
# See Definition 4 in the paper.
HIGHER_ORDER_IMPORTANCE_K = 100

# The number of repetitions for data exploration.
N_EXPLORATION = 1

# If set to True it will scale the fingerprints
FINGERPRINT_SCALING = True