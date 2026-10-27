"""Group trucks by where they actually run.

Ported from notebooks/original/01_movement_similarity_clusters.ipynb,
02_two_layer_dbscan.ipynb and 03_region_movement_profile.ipynb.

Input: one row per vehicle with its top-10 most visited location clusters
(cluster1..cluster10) and the share of time spent in each (priority1..priority10).

1. one-hot the visited clusters into a vehicle x cluster matrix
2. layer 1: DBSCAN across all vehicles (trucks with near-identical footprints group together)
3. layer 2: DBSCAN again inside each layer-1 group to split it further
4. region profile: add up time share per region and tag the vehicle with every region
   above 20% (e.g. "NORTH + WEST" for a Delhi-Mumbai lane)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

REGIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL', 'NORTH_EAST']


def one_hot(movement, n_clusters):
    final_data = []
    for _, row in movement.fillna(-1).iterrows():
        new_row = [0] * (n_clusters + 1)
        new_row[0] = row['vehicle_number']
        for k in range(1, 11):
            c = int(row[f'cluster{k}'])
            if c > 0:
                new_row[c] = 1
        final_data.append(new_row)
    cols = ['vehicle_number'] + [f'cluster_{i}' for i in range(1, n_clusters + 1)]
    df = pd.DataFrame(final_data, columns=cols).drop_duplicates('vehicle_number')
    keep = ['vehicle_number'] + [c for c in cols[1:] if df[c].sum() > 0]  # drop unused clusters
    return df[keep]


def layer1(onehot, eps=2, min_samples=2):
    x = onehot.drop('vehicle_number', axis=1).values
    out = onehot.copy()
    out['DB_clustered'] = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(x).labels_
    return out


def layer2(layer1_df, eps=2, min_samples=5):
    """Same loop as the original: split by layer-1 label, drop empty columns, DBSCAN again."""
    parts = []
    for lab in layer1_df.DB_clustered.unique():
        part = layer1_df[layer1_df['DB_clustered'] == lab].copy()
        feats = part.drop(['vehicle_number', 'DB_clustered'], axis=1)
        feats = feats.loc[:, feats.sum() > 0]
        if lab == -1 or len(part) < min_samples:
            part['DB_clustered_layer2'] = -1
        else:
            part['DB_clustered_layer2'] = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(feats.values).labels_
        parts.append(part[['vehicle_number', 'DB_clustered', 'DB_clustered_layer2']])
    return pd.concat(parts)


def kmeans_baseline(onehot, k):
    x = onehot.drop('vehicle_number', axis=1).values
    return KMeans(n_clusters=k, random_state=0, n_init=10).fit(x).labels_


def region_movement(movement, clusters, threshold=20):
    """clusters: cluster id -> state, region."""
    m = movement.fillna(-1).copy()
    lookup = clusters.set_index('clusters')['region'].to_dict()
    regions = []
    for _, row in m.iterrows():
        tot = dict.fromkeys(REGIONS, 0)
        for i in range(1, 11):
            c = row[f'cluster{i}']
            if c <= -1:
                break
            r = str(lookup.get(c, '')).upper().replace(' ', '_')
            if r in tot:
                tot[r] += row[f'priority{i}']
        regions.append(' + '.join(r for r in REGIONS if tot[r] > threshold))
    m['region_movement'] = regions
    return m[['vehicle_number', 'region_movement']]
