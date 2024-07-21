### get the fingerprints
# read prepared explaining features data sheet
# draw gap statistic plot (based on config)
# based on config, apply clustering
# save the response types (observe that this are still scaled variables!)
import os.path

import pandas as pd
from gapstatistics.gapstatistics import GapStatistics
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np

import config


# calculate fingerprints per group, use oversampled=False dataframe
# draw dendrogram and save the pairwise distances
## output -> excel with fingerprints, response types, dendrogram, pairwise distances

class ClusteringApplier:

    @staticmethod
    def read_observable_data() -> pd.DataFrame|None:
        required_file: str = f"{config.OUTPUT_FOLDER_BASE}base_data/{config.DATASET_NAME}_observable_dataset_scaled.xlsx"
        try:
            df: pd.DataFrame = pd.read_excel(required_file)
            df = df[list(config.OBSERVABLE_FEATURE_NAMES.keys())]
        except:
            print(f"Error. Invalid file {required_file}.")
            return None

        return df

    @staticmethod
    def draw_gap_statistic_plot() -> None:

        df: pd.DataFrame = ClusteringApplier.read_observable_data()

        gs = GapStatistics(algorithm=AgglomerativeClustering, distance_metric='minkowski', return_params=True)
        num_colors: int = config.GAP_STATISTIC_CLUSTER_RANGE
        cm = plt.get_cmap('gist_rainbow')
        color_dict = {i: cm(1.*i/num_colors) for i in range(num_colors)}
        optimum, params = gs.fit_predict(K=config.GAP_STATISTIC_CLUSTER_RANGE, X=df.to_numpy())
        gs.plot(original_labels=df.columns, colors=color_dict)
        plt.savefig(f"{config.OUTPUT_FOLDER_BASE}gapstat/{config.DATASET_NAME}_gap-statistic-plot.pdf", bbox_inches='tight')

    @staticmethod
    def do_gap_statistic(max_clust=20):
        def clustering_method(X, k):
            # Adjust here, if not ward's clustering algorithm should be taken
            ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            clf = NearestCentroid()
            clf.fit(X, ward)
            return clf.centroids_, ward

        df: pd.DataFrame = pd.read_excel('/mnt/f/Nextcloud/Dierkes/Naturverbundenheit/Fragebogenclustering_V2_imputed.xlsx')
        df_use = df[
            ['Pollution', 'Climate change', 'Invasive species', 'Habitat loss', 'Overexploitation', 'Discrimination']]
        df_use = df_use.rename(columns=COLUMNRENAME)

        rcParams['figure.figsize'] = 18.7 / 2, 8.27 / 2

        optimalK = OptimalK(clusterer=clustering_method, n_iter=20)

        noise = np.random.normal(loc=0.0, scale=0.2, size=[len(df_use.index), 6])
        df_use += noise

        n_clusters = optimalK(df_use, cluster_array=range(2, max_clust))

        plt.cla()
        plt.clf()

        plt.errorbar(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, xerr=optimalK.gap_df.sk, marker='o',
                     color='black')

        plt.title('Gap statistic')
        plt.xlabel('Number of Clusters')
        plt.xticks(ticks=list(range(2, max_clust)))
        plt.ylabel('Gap')

        ax = plt.gca()
        ax.grid(which='major', axis='x', linestyle='--')

        optimalK.gap_df.to_excel(
            '/mnt/f/Nextcloud/Dierkes/Naturverbundenheit/revision/drei_disruptoren/gap_statistic1.xlsx', index=False)
        plt.savefig('/mnt/f/Nextcloud/Dierkes/Naturverbundenheit/revision/drei_disruptoren/gap_statistic1.pdf', dpi=300,
                    bbox_inches='tight')

