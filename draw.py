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
    # for edge in edges:
    #     x, y = edge.xy
    #     ax.plot(x, y, color='grey', linewidth=0.5, linestyle='--')

    ax.set_aspect('equal')
    ax.set_title("Polygonized City Blocks")
    plt.show()

def plot_unused_edges(edges):
    # Build set of all edges used in polygons
    used_edges = set()

    for poly in polygonize(edges):
        coords = list(poly.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            line = LineString([a, b])
            used_edges.add(line)

    # Now identify unused edges
    unused_edges = [e for e in edges if not any(e.equals(ue) for ue in used_edges)]
    print(unused_edges)
    fig, ax = plt.subplots(figsize=(10, 10))
    for edge in unused_edges:
        x, y = edge.xy
        ax.plot(x, y, color='red', linewidth=1)
    ax.set_aspect('equal')
    plt.show()

def split_edges_by_polygonizability(edges):
    """Return (used_edges, unused_edges) from an edge list."""
    used_edges_set = set()
    polygons = list(polygonize(edges))

    for poly in polygons:
        coords = list(poly.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            used_edges_set.add(LineString([a, b]))

    # Separate edges based on whether they're used in any polygon
    used_edges = []
    unused_edges = []
    for e in edges:
        if any(e.equals(ue) for ue in used_edges_set):
            used_edges.append(e)
        else:
            unused_edges.append(e)

    return polygons, used_edges, unused_edges

import networkx as nx

def extract_cycles_from_edges(edges):
    """Find simple cycles in the set of edges using NetworkX."""
    G = nx.Graph()

    for edge in edges:
        coords = list(edge.coords)
        a = tuple(coords[0])
        b = tuple(coords[1])
        G.add_edge(a, b)

    # Extract simple cycles (up to some length)
    # NOTE: This is a generator, so you can limit depth or number
    cycles = list(nx.cycle_basis(G))

    # Convert to LineStrings
    polygons = []
    for cycle in cycles:
        if len(cycle) >= 3:
            # Close the loop explicitly
            cycle.append(cycle[0])
            polygons.append(LineString(cycle))

    return polygons

edges_with_ids = build_edges_from_streets(streets)
edges = [e for e, _ in edges_with_ids]

polygons, used_edges, orphaned_edges = split_edges_by_polygonizability(edges)
# Find extra loops from orphaned edges
wild_polygons = extract_cycles_from_edges(orphaned_edges)

fig, ax = plt.subplots(figsize=(10, 10))

# Main polygons
for poly in polygons:
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.5, edgecolor='black',facecolor='lightblue')

# Wild ones
for loop in wild_polygons:
    x, y = loop.xy
    ax.fill(x, y, alpha=0.5, edgecolor='blue', facecolor='lightgreen')
    # ax.plot(x, y, color='red', linestyle='--', linewidth=1)

# Original edges (if you want)
# for edge in edges:
#     x, y = edge.xy
#     ax.plot(x, y, color='grey', linewidth=0.3, linestyle=':')

ax.set_aspect('equal')
plt.title("City Blocks + Wild Polygons")
plt.show()