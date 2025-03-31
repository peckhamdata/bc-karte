from json import load
from shapely.geometry import LineString, Polygon

def load_city_model(filepath):
    with open(filepath, "r") as f:
        raw = load(f)

    # Rebuild geometry objects
    blocks = [
        {
            "id": blk["id"],
            "polygon": Polygon(blk["polygon"]),
            "edge_ids": blk["edge_ids"],
            "street_ids": blk["street_ids"]
        }
        for blk in raw["blocks"]
    ]

    edges = [
        {
            "id": edge["id"],
            "geometry": LineString(edge["geometry"]),
            "street_id": edge["street_id"],
            "junction_ids": edge["junction_ids"]
        }
        for edge in raw["edges"]
    ]

    streets = raw["streets"]  # already JSON-serializable
    junctions = raw["junctions"]  # likewise

    return {
        "blocks": blocks,
        "edges": edges,
        "streets": streets,
        "junctions": junctions
    }

def walk_blocks(city_model, block_ids=None):
    """Return list of edge dicts used by one or more blocks."""
    edges = []
    block_list = city_model["blocks"]

    if block_ids is not None:
        block_list = [b for b in block_list if b["id"] in block_ids]

    seen_edges = set()
    for block in block_list:
        for eid in block["edge_ids"]:
            if eid not in seen_edges:
                edge = next(e for e in city_model["edges"] if e["id"] == eid)
                edges.append(edge)
                seen_edges.add(eid)

    return edges

def walk_street(city_model, street_id):
    """Return list of edge dicts for a given street."""
    street = next(s for s in city_model["streets"] if s["id"] == street_id)
    return [e for e in city_model["edges"] if e["id"] in street["edge_ids"]]


import random

def walk_random_junctions(city_model, steps=10):
    """Take a random walk from junction to junction, return list of edges used."""
    current_junc = random.choice(city_model["junctions"])
    walked_edges = []

    for _ in range(steps):
        connected = current_junc["edge_ids"]
        available = [eid for eid in connected if eid not in [e["id"] for e in walked_edges]]
        if not available:
            break

        eid = random.choice(available)
        edge = next(e for e in city_model["edges"] if e["id"] == eid)
        walked_edges.append(edge)

        # Move to next junction
        next_junc_id = [j for j in edge["junction_ids"] if j != current_junc["id"]]
        if not next_junc_id:
            break
        current_junc = next(j for j in city_model["junctions"] if j["id"] == next_junc_id[0])

    return walked_edges

import matplotlib.pyplot as plt

def plot_walked_edges(edges, ax=None, color="dodgerblue", title="Walk View"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    for edge in edges:
        x, y = edge["geometry"].xy
        ax.plot(x, y, color=color, linewidth=2)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.show()


city_model = load_city_model("bezier_city_model.json")

# Pick a block
walked = walk_blocks(city_model, block_ids=[0])
plot_walked_edges(walked, title="Block Walk")

# Or pick a street
walked = walk_street(city_model, street_id=1)
plot_walked_edges(walked, title="Street 17")

# Or take a meandering journey
walked = walk_random_junctions(city_model, steps=25)
plot_walked_edges(walked, title="Random Junction Walk")