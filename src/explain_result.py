### feature selection
# read observable pairwise distances and scale it to 1
# either greedy or all subset-based:
## greedy (given that already n features have been added)
### from all not added features, add one, calculate pairwise distances and scale it to 1
### apply small distortion
### make linear regression and save r^2 to a list
### chose as n+1 feature the one which maximised r^2
## subset based: same but for all 2^n combinations
# save the optimal features -> excel

### visualization
# read optimal features and the corresponding normalized pairwise distances
# read observable features and the corresponding normalized distances
# make linear regression and output -> regression plot, excel with used distances (both matrices) after scaling and distorting
# make radar plots:
# - fingerprint per group
# - explaining plot per group based on the optimal feature set (scaled)
# - obserable patterns (scaled)
# - linear regression plot

import pandas as pd

import config


class ResultExplainer:

    pass