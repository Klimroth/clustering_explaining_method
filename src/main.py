from data_preparation import DataPreparator
from apply_clustering import ClusteringApplier

CONDUCT_DATA_PREPARATION = True
CONDUCT_GAP_STAT_ANALYSIS = True
CONDUCT_OBSERVABLE_CLUSTERING = True
CONDUCT_EXPLAINABLE_DISTANCES = True


if CONDUCT_DATA_PREPARATION:
    DataPreparator.prepare_data()

if CONDUCT_GAP_STAT_ANALYSIS:
    ClusteringApplier.draw_gap_statistic_plot()

if CONDUCT_OBSERVABLE_CLUSTERING:
    ClusteringApplier.calculate_observable_patterns()

if CONDUCT_EXPLAINABLE_DISTANCES:
    ClusteringApplier.calculate_explainable_distances()
