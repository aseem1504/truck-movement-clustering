"""Find maintenance hubs: places where trucks sit still right before the battery is pulled.

Ported from notebooks/original/04_maintenance_hub_hdbscan.ipynb.
Take the hour of pings before every BATTERY_DISCONNECTED event, keep the slow ones
(speed < 5), and run HDBSCAN with the haversine metric. Dense clusters are garages and
workshops where batteries get disconnected for repair (or where tampering happens).

Fix vs original: haversine in HDBSCAN expects [lat, lon] in radians. The original computed
radians but then fitted on raw [lon, lat] degrees, which distorts distances. Fixed here.
"""
import numpy as np
import pandas as pd

EARTH_KM = 6371.0


def find_hubs(pings, min_cluster_size=15, min_vehicles=5):
    import hdbscan
    p = pings[pings['speed'] < 5].copy()
    lonlat = np.array(p['location'].tolist())
    rads = np.radians(lonlat[:, ::-1])  # -> [lat, lon]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine').fit(rads)
    p['cluster'] = clusterer.labels_
    p['probabilities'] = clusterer.probabilities_
    p['longitude'], p['latitude'] = lonlat[:, 0], lonlat[:, 1]
    hubs = (p[p.cluster >= 0].groupby('cluster')
            .agg(latitude=('latitude', 'median'), longitude=('longitude', 'median'),
                 pings=('time', 'size'), vehicles=('vehicleId', 'nunique'))
            .sort_values('vehicles', ascending=False).reset_index())
    # one truck parked for a long time is not a hub; ask for several distinct trucks
    hubs = hubs[hubs.vehicles >= min_vehicles].reset_index(drop=True)
    return hubs, p


def india_grid(n_rows=335, n_cols=288):
    """Layer-1 grid over India's bounding box (lat 8-38, lon 68-98), from
    notebooks/original/05_india_grid_reverse_geocode.ipynb. Layer 2 merges 5x5 cells,
    layer 3 merges 2x2 layer-2 cells, same as the original."""
    rows1 = np.linspace(8.0, 38.0, num=n_rows)
    cols1 = np.linspace(68.0, 98.0, num=n_cols)
    g = pd.DataFrame([(r + 1, c + 1, lat, lat + 30 / n_rows, lon, lon + 30 / n_cols)
                      for r, lat in enumerate(rows1) for c, lon in enumerate(cols1)],
                     columns=['layer1_row', 'layer1_col', 'from_lat', 'to_lat', 'from_lon', 'to_lon'])
    g.insert(0, 'layer1', range(1, len(g) + 1))
    g['layer2_row'] = np.searchsorted(list(range(0, n_rows, 5)), g['layer1_row'])
    g['layer2_col'] = np.searchsorted(list(range(0, 291, 5)), g['layer1_col'])
    g['layer2'] = (g['layer2_row'] - 1) * 58 + g['layer2_col']
    g['layer3_row'] = np.searchsorted(list(range(0, 68, 2)), g['layer2_row'])
    g['layer3_col'] = np.searchsorted(list(range(0, 59, 2)), g['layer2_col'])
    g['layer3'] = (g['layer3_row'] - 1) * 29 + g['layer3_col']
    return g


def cell_of(grid, lat, lon):
    a = grid[(grid.from_lat <= lat) & (grid.to_lat > lat) & (grid.from_lon <= lon) & (grid.to_lon > lon)]
    return None if a.empty else {k: int(v) for k, v in a.iloc[0][['layer1', 'layer2', 'layer3']].items()}
