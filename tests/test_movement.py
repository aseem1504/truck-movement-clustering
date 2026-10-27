import pandas as pd

from movement import simulate
from movement.clustering import layer1, one_hot, region_movement
from movement.hubs import cell_of, india_grid

HUBS = pd.read_csv('data/freight_hubs.csv')


def test_one_hot_and_layer1():
    fp, ct = simulate.footprints(HUBS, n_vehicles=200)
    oh = one_hot(fp, ct.clusters.max())
    assert oh.drop(columns='vehicle_number').values.max() == 1
    assert 'DB_clustered' in layer1(oh)


def test_region_profile():
    row = {'vehicle_number': 'A', 'cluster1': 1, 'priority1': 60, 'cluster2': 2, 'priority2': 40}
    for i in range(3, 11):
        row[f'cluster{i}'] = None
        row[f'priority{i}'] = None
    ct = pd.DataFrame({'clusters': [1, 2], 'region': ['North', 'West']})
    assert region_movement(pd.DataFrame([row]), ct).iloc[0].region_movement == 'NORTH + WEST'


def test_grid_lookup():
    g = india_grid()
    assert cell_of(g, 28.61, 77.21)['layer1'] > 0
