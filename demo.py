"""Demo on synthetic data: python demo.py

1. cluster trucks by footprint (two-layer DBSCAN) and check against their hidden lane
2. tag each truck with the regions it runs in
3. find maintenance hubs from pre-disconnect pings (HDBSCAN)
"""
import os

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from movement import simulate
from movement.clustering import kmeans_baseline, layer1, layer2, one_hot, region_movement
from movement.hubs import cell_of, find_hubs, india_grid

pd.set_option('display.width', 200)
os.makedirs('outputs', exist_ok=True)
hubs = pd.read_csv('data/freight_hubs.csv')

print("=== 1. Movement similarity clustering ===")
fp, clusters = simulate.footprints(hubs)
oh = one_hot(fp, clusters.clusters.max())
print(f"{len(oh)} trucks x {oh.shape[1] - 1} visited location clusters")
l1 = layer1(oh)
l2 = layer2(l1)
res = fp[['vehicle_number', 'true_lane']].merge(l2, on='vehicle_number')
res['two_layer'] = res.DB_clustered.astype(str) + '_' + res.DB_clustered_layer2.astype(str)
noise = (res.DB_clustered == -1).mean()
print(f"layer 1 groups: {res.DB_clustered.nunique() - (noise > 0)}, left as noise: {noise:.0%}")
print(f"layer 2 groups: {res[res.DB_clustered_layer2 >= 0].two_layer.nunique()}")
core = res[res.DB_clustered >= 0]
print(f"agreement with true lane (ARI, 1 = perfect): layer1 {adjusted_rand_score(core.true_lane, core.DB_clustered):.2f}"
      f" | k-means (k=10) on all trucks {adjusted_rand_score(fp.true_lane, kmeans_baseline(oh, 10)):.2f}")
print("\nLargest layer-1 groups and the lane they mostly are:")
top = core.groupby('DB_clustered').agg(trucks=('vehicle_number', 'size'),
                                        main_lane=('true_lane', lambda s: s.value_counts().index[0]),
                                        purity=('true_lane', lambda s: round(s.value_counts(normalize=True).iloc[0], 2)))
print(top.sort_values('trucks', ascending=False).head(8).to_string())
res.to_csv('outputs/movement_clusters.csv', index=False)

print("\n=== 2. Region movement profile ===")
reg = region_movement(fp, clusters).merge(fp[['vehicle_number', 'true_lane']])
print(reg.groupby('true_lane').region_movement.agg(lambda s: s.value_counts().index[0]).to_string())
reg.to_csv('outputs/region_movement.csv', index=False)

print("\n=== 3. Maintenance hubs from pre-disconnect pings ===")
pings, garages = simulate.disconnect_pings(hubs)
found, _ = find_hubs(pings)
grid = india_grid()
found['grid_cells'] = found.apply(lambda r: cell_of(grid, r.latitude, r.longitude), axis=1)
from movement.geo import haversine
found['nearest_true_workshop_m'] = found.apply(lambda r: min(haversine([r.longitude, r.latitude], [g.lon, g.lat])
                                                             for g in garages.itertuples()), axis=1).round(0)
print(f"true workshops: {len(garages)}  hubs found (5+ trucks): {len(found)}")
print(found.head(10).to_string(index=False))
found.to_csv('outputs/maintenance_hubs.csv', index=False)
