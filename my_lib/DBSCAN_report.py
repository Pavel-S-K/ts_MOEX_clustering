import pandas as pd

pd.set_option("display.max_columns", 50)


def DBSCAN_report(DBSCAN_results: pd.DataFrame, sil_round_val: int = 1):
    silhouette_report = DBSCAN_results.sort_values(by='silhouette', ascending=False)['silhouette'].round(
        sil_round_val).value_counts()
    silhouette_report = pd.DataFrame(silhouette_report)

    clusters_report = DBSCAN_results.sort_values(by='clusters', ascending=False)['clusters'].value_counts()
    clusters_report = pd.DataFrame(clusters_report)

    noise_report = DBSCAN_results.sort_values(by='noise', ascending=False)['noise'].value_counts()
    noise_report = pd.DataFrame(noise_report)

    print('silhouette_report:')
    print(silhouette_report)

    print('\n')
    print('clusters_report:')
    print(clusters_report)

    print('\n')
    print('noise_report:')
    print(noise_report)
