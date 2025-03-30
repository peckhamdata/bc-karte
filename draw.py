import matplotlib.pyplot as plt
from json import loads
import random
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize

# Load your street data
with open("bezier_city.json", "r") as f:
    city_data = loads(f.read())

streets = city_data["streets"]

def random_rgb_255():
    return (random.random(), random.random(), random.random())


def build_edges_from_streets(streets):
    edges_with_ids = []

    for street in streets:
        juncs = street['junctions']
        street_id = street['id']

        for a, b in zip(juncs[:-1], juncs[1:]):
            pt_a = (a['x'], a['y'])
            pt_b = (b['x'], b['y'])

            if pt_a == pt_b:
                continue

            edge = LineString([pt_a, pt_b])
            edges_with_ids.append((edge, street_id))

    return edges_with_ids

def polygonize_and_plot(edges):
    # Polygonize
    polygons = list(polygonize(edges))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, edgecolor='black', linewidth=1)

    # Optionally plot edges too
    for edge in edges:
        x, y = edge.xy
        ax.plot(x, y, color='grey', linewidth=0.5, linestyle='--')

    ax.set_aspect('equal')
    ax.set_title("Polygonized City Blocks")
    plt.show()

# 👇 Run this after sorting junctions

from shapely.strtree import STRtree

edges_with_ids = build_edges_from_streets(streets)
edges = [e for e, _ in edges_with_ids]
tree = STRtree(edges)
edge_to_id = {id(e): sid for e, sid in edges_with_ids}

polygonize_and_plot(edges)
