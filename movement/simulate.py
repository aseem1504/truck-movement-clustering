"""Synthetic fleet footprints and pre-disconnect pings.

Hub cities and their coordinates are public (data/freight_hubs.csv). Every truck,
lane and ping is generated here. Each city is split into 3 location clusters so the
one-hot matrix looks like the original (a few hundred clusters, sparse rows).
"""
import numpy as np
import pandas as pd

LANES = {  # lane name -> cities, used as the hidden "true" operating pattern
    'delhi_mumbai': ['Delhi', 'Jaipur', 'Ahmedabad', 'Vadodara', 'Surat', 'Mumbai'],
    'delhi_kolkata': ['Delhi', 'Agra', 'Kanpur', 'Lucknow', 'Patna', 'Dhanbad', 'Kolkata'],
    'chennai_bengaluru': ['Chennai', 'Bengaluru', 'Coimbatore'],
    'mumbai_pune_nashik': ['Mumbai', 'Pune', 'Nashik'],
    'gujarat_ports': ['Gandhidham', 'Ahmedabad', 'Vadodara'],
    'hyderabad_vijayawada': ['Hyderabad', 'Vijayawada', 'Visakhapatnam'],
    'central_belt': ['Nagpur', 'Raipur', 'Jabalpur', 'Bhopal', 'Indore'],
    'punjab_ncr': ['Ludhiana', 'Panipat', 'Delhi'],
    'east_steel': ['Jamshedpur', 'Dhanbad', 'Kolkata', 'Bhubaneswar'],
    'northeast_feeder': ['Kolkata', 'Guwahati'],
}


def cluster_table(hubs):
    rows, cid = [], 1
    for h in hubs.itertuples():
        for _ in range(3):
            rows.append({'clusters': cid, 'city': h.city, 'state': h.state, 'region': h.region})
            cid += 1
    return pd.DataFrame(rows)


def footprints(hubs, n_vehicles=1200, seed=21):
    rng = np.random.default_rng(seed)
    ct = cluster_table(hubs)
    by_city = ct.groupby('city').clusters.apply(list).to_dict()
    lanes = list(LANES)
    weights = np.array([.2, .12, .12, .14, .08, .08, .08, .08, .06, .04])
    rows = []
    for k in range(n_vehicles):
        lane = rng.choice(lanes, p=weights)
        cities = LANES[lane]
        visited = []
        for c in cities:  # each truck uses 1-2 of the 3 clusters in every city on its lane
            visited += list(rng.choice(by_city[c], size=rng.integers(1, 3), replace=False))
        if rng.random() < 0.15:  # odd trips outside the lane
            visited += list(rng.choice(ct.clusters, size=2, replace=False))
        visited = list(dict.fromkeys(visited))[:10]
        share = rng.dirichlet(np.ones(len(visited))) * 100
        order = np.argsort(-share)
        row = {'vehicle_number': f'SIM{k:05d}', 'true_lane': lane}
        for i in range(10):
            row[f'cluster{i + 1}'] = int(visited[order[i]]) if i < len(visited) else np.nan
            row[f'priority{i + 1}'] = round(float(share[order[i]]), 1) if i < len(visited) else np.nan
        rows.append(row)
    return pd.DataFrame(rows), ct


def disconnect_pings(hubs, n_garages=8, seed=4):
    """Slow pings in the hour before BATTERY_DISCONNECTED events: most happen at a few
    workshops, the rest are scattered roadside stops."""
    rng = np.random.default_rng(seed)
    garages = hubs.sample(n_garages, random_state=seed)[['city', 'lat', 'lon']].copy()
    garages['lat'] += rng.normal(0, 0.03, n_garages)
    garages['lon'] += rng.normal(0, 0.03, n_garages)
    rows, t = [], 1_600_000_000
    for v in range(600):
        if rng.random() < 0.7:
            g = garages.iloc[rng.integers(0, n_garages)]
            lat, lon = g.lat + rng.normal(0, 0.0015), g.lon + rng.normal(0, 0.0015)
        else:
            lat, lon = rng.uniform(10, 30), rng.uniform(72, 88)
        for i in range(rng.integers(5, 20)):
            rows.append({'vehicleId': 400000 + v, 'time': t + i * 60,
                         'location': [lon + rng.normal(0, 2e-4), lat + rng.normal(0, 2e-4)],
                         'speed': abs(rng.normal(1, 1.5))})
        t += 3600
    return pd.DataFrame(rows), garages.reset_index(drop=True)
