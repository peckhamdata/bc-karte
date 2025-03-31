from collections import defaultdict
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
import networkx as nx
from json import loads, dump

from collections import defaultdict
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
import networkx as nx
import time

def generate_city_model(streets):
    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    log("🔄 Starting city model generation")

    # Step 1: Extract edges from sorted junctions
    log("📦 Extracting edges from streets...")
    edges_with_ids = []
    for street in streets:
        juncs = street['junctions']
        sid = street['id']
        for a, b in zip(juncs[:-1], juncs[1:]):
            pt_a, pt_b = (a['x'], a['y']), (b['x'], b['y'])
            if pt_a != pt_b:
                edge = LineString([pt_a, pt_b])
                edges_with_ids.append((edge, sid))
    log(f"✅ {len(edges_with_ids)} edges extracted.")

    # Step 2: Assign edge IDs
    log("🧷 Assigning edge IDs...")
    edges_data = []
    edge_id_lookup = {}
    for idx, (edge, street_id) in enumerate(edges_with_ids):
        coords = list(edge.coords)
        eid = idx
        edges_data.append({
            "id": eid,
            "geometry": edge,
            "street_id": street_id,
            "junction_coords": [tuple(coords[0]), tuple(coords[1])]
        })
        edge_id_lookup[id(edge)] = eid

    # Step 3: Extract junctions
    log("🔗 Building junctions...")
    junction_map = defaultdict(set)
    for edge in edges_data:
        for pt in edge["junction_coords"]:
            junction_map[pt].add(edge["id"])

    junctions_data = []
    junction_id_lookup = {}
    for idx, (pt, edge_ids) in enumerate(junction_map.items()):
        junctions_data.append({
            "id": idx,
            "coords": pt,
            "edge_ids": list(edge_ids)
        })
        junction_id_lookup[pt] = idx

    for edge in edges_data:
        edge["junction_ids"] = [
            junction_id_lookup[edge["junction_coords"][0]],
            junction_id_lookup[edge["junction_coords"][1]],
        ]

    log(f"✅ {len(junctions_data)} junctions created.")

    # Step 4: Group into streets
    log("🧭 Grouping edges into streets...")
    street_edges = defaultdict(list)
    for edge in edges_data:
        street_edges[edge["street_id"]].append(edge["id"])

    streets_data = [{"id": sid, "edge_ids": edge_ids} for sid, edge_ids in street_edges.items()]
    log(f"✅ {len(streets_data)} streets grouped.")

    # Step 5: Polygonize
    log("🧱 Polygonizing edge network...")
    edges_geom = [e["geometry"] for e in edges_data]
    polygons = list(polygonize(edges_geom))
    log(f"✅ {len(polygons)} regular polygons found.")

    # Step 6: Orphaned edges
    used_edges_set = set()
    for poly in polygons:
        coords = list(poly.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            used_edges_set.add(LineString([a, b]))

    orphaned_edges = [e for e in edges_geom if not any(e.equals(ue) for ue in used_edges_set)]
    log(f"🕸️ {len(orphaned_edges)} orphaned edges identified.")

    # Step 7: Extract wild polygons
    def extract_cycles_from_edges(edges):
        G = nx.Graph()
        for edge in edges:
            a, b = map(tuple, edge.coords)
            G.add_edge(a, b)
        cycles = list(nx.cycle_basis(G))
        loops = []
        for c in cycles:
            if len(c) >= 3:
                c.append(c[0])
                loops.append(Polygon(c))
        return loops

    log("🌪️ Extracting wild polygons from orphaned edges...")
    wild_polygons = extract_cycles_from_edges(orphaned_edges)
    log(f"✅ {len(wild_polygons)} wild polygons created.")

    # Step 8: Build blocks from polygons
    def build_blocks_from_polygons(poly_list, label_offset=0):
        blocks = []
        current_id = label_offset
        for poly in poly_list:
            coords = list(poly.exterior.coords)
            edge_ids = []
            for a, b in zip(coords[:-1], coords[1:]):
                candidate = LineString([a, b])
                for edge in edges_data:
                    geom = edge["geometry"]
                    if candidate.equals(geom) or candidate.equals(LineString(geom.coords[::-1])):
                        edge_ids.append(edge["id"])
                        break
            street_ids = list(set(edge["street_id"] for edge in edges_data if edge["id"] in edge_ids))
            blocks.append({
                "id": current_id,
                "polygon": poly,
                "edge_ids": edge_ids,
                "street_ids": street_ids
            })
            current_id += 1
        return blocks

    log("🧩 Building regular blocks...")
    blocks_regular = build_blocks_from_polygons(polygons)
    log("🧩 Building wild blocks...")
    blocks_wild = build_blocks_from_polygons(wild_polygons, label_offset=len(blocks_regular))
    all_blocks = blocks_regular + blocks_wild
    log(f"✅ Total blocks: {len(all_blocks)}")

    # Final output
    log("📦 Assembling city model...")
    city_model = {
        "blocks": all_blocks,
        "edges": edges_data,
        "streets": streets_data,
        "junctions": junctions_data
    }

    log("✅ City model generation complete.")
    return city_model


def serialize_city_model(city_model, filepath):

    def geom_to_coords(geom):
        if isinstance(geom, LineString):
            return list(geom.coords)
        elif isinstance(geom, Polygon):
            return list(geom.exterior.coords)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geom)}")

    output = {
        "blocks": [
            {
                "id": blk["id"],
                "polygon": geom_to_coords(blk["polygon"]),
                "edge_ids": blk["edge_ids"],
                "street_ids": blk["street_ids"]
            }
            for blk in city_model["blocks"]
        ],
        "edges": [
            {
                "id": edge["id"],
                "geometry": list(edge["geometry"].coords),
                "street_id": edge["street_id"],
                "junction_ids": edge["junction_ids"]
            }
            for edge in city_model["edges"]
        ],
        "streets": city_model["streets"],  # already serializable
        "junctions": [
            {
                "id": junc["id"],
                "coords": junc["coords"],
                "edge_ids": junc["edge_ids"]
            }
            for junc in city_model["junctions"]
        ]
    }

    with open(filepath, "w") as f:
        dump(output, f, indent=2)


# Load your street data
with open("bezier_city.json", "r") as f:
    city_data = loads(f.read())

streets = city_data["streets"]

city_model = generate_city_model(streets)

serialize_city_model(city_model, "bezier_city_model.json")
