import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point, box
from scipy.spatial import Voronoi
import random

# Load blocks
with open('bezier_city_model.json', 'r') as f:
    data = json.load(f)

blocks = data['blocks']

# Helper to generate random points inside a polygon
def random_points_in_polygon(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(p):
            points.append(p)
    return points

# Subdivide a polygon into rectangles
def split_polygon_into_rectangles(poly, rows, cols):
    minx, miny, maxx, maxy = poly.bounds
    width = (maxx - minx) / cols
    height = (maxy - miny) / rows
    rectangles = []
    
    for i in range(cols):
        for j in range(rows):
            rect = box(minx + i * width, miny + j * height,
                       minx + (i + 1) * width, miny + (j + 1) * height)
            
            # ADD: Plot the raw grid if you want to see it
            # patch = plt.Polygon(list(rect.exterior.coords), edgecolor='gray', fill=False)
            # ax.add_patch(patch)
            
            clipped = poly.intersection(rect)
            if not clipped.is_empty and clipped.area > 1e-6:  # Tiny slivers get ignored
                if isinstance(clipped, (Polygon, MultiPolygon)):
                    rectangles.append(clipped)
                    
    print(f"Created {len(rectangles)} cells for one polygon")
    return rectangles


# Subdivide a polygon using Voronoi
def split_polygon_into_voronoi(poly, num_points):
    seeds = random_points_in_polygon(poly, num_points)
    seed_coords = np.array([[p.x, p.y] for p in seeds])
    vor = Voronoi(seed_coords)
    
    cells = []
    for region_idx in vor.point_region:
        vertices = vor.regions[region_idx]
        if -1 not in vertices and vertices:
            region = [vor.vertices[i] for i in vertices]
            cell = Polygon(region)
            if not cell.is_valid:
                cell = cell.buffer(0)
            if not poly.is_valid:
                poly = poly.buffer(0)
            clipped_cell = cell.intersection(poly)
            if not clipped_cell.is_empty:
                if isinstance(clipped_cell, (Polygon, MultiPolygon)):
                    cells.append(clipped_cell)
    return cells


def bsp_split(poly, depth=3):
    """Recursively split a polygon using Binary Space Partitioning (BSP)."""
    if depth == 0:
        return [poly]

    minx, miny, maxx, maxy = poly.bounds
    if random.random() > 0.5:  # Randomly choose to split vertically or horizontally
        split_x = random.uniform(minx + 0.3 * (maxx - minx), maxx - 0.3 * (maxx - minx))
        left = box(minx, miny, split_x, maxy)
        right = box(split_x, miny, maxx, maxy)
        parts = (left, right)
    else:  # Horizontal split
        split_y = random.uniform(miny + 0.3 * (maxy - miny), maxy - 0.3 * (maxy - miny))
        bottom = box(minx, miny, maxx, split_y)
        top = box(minx, split_y, maxx, maxy)
        parts = (bottom, top)

    # Recursively split each part
    result = []
    for part in parts:
        clipped = poly.intersection(part)
        if not clipped.is_empty:
            result.extend(bsp_split(clipped, depth - 1))
    return result

import math

def create_hex_grid(poly, hex_size):
    """Create a hexagonal grid clipped to a polygon."""
    minx, miny, maxx, maxy = poly.bounds
    dx = 3/2 * hex_size
    dy = math.sqrt(3) * hex_size
    
    cells = []
    
    x = minx
    while x < maxx + hex_size:
        y = miny
        while y < maxy + hex_size:
            # Offset every other column ("odd-r" layout)
            offset = 0 if int((x - minx) / dx) % 2 == 0 else dy / 2
            
            hex_center = (x, y + offset)
            hexagon = create_hexagon(hex_center, hex_size)
            clipped = poly.intersection(hexagon)
            if not clipped.is_empty:
                cells.append(clipped)
            y += dy
        x += dx
    
    return cells

def create_hexagon(center, size):
    """Create a regular hexagon polygon centered at a point."""
    cx, cy = center
    angles = [math.radians(a) for a in range(0, 360, 60)]
    points = [(cx + size * math.cos(a), cy + size * math.sin(a)) for a in angles]
    return Polygon(points)

# At top of function (initialize a global cell counter)
cell_counter = 0

def subdivide_blocks(blocks, use_methods, **kwargs):
    global cell_counter  # <- allow modification
    for idx, block in enumerate(blocks):
        coords = block['polygon']
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        methods = ['rectangles', 'voronoi', 'bsp', 'hex']
        method = methods[use_methods[idx] - 1]
        print(f"Using method {method} for block {idx}")
        if method == 'rectangles':
            rows = kwargs.get('rows')[idx]
            cols = kwargs.get('cols')[idx]
            cells = split_polygon_into_rectangles(poly, rows, cols)
        elif method == 'voronoi':
            num_points = kwargs.get('num_points', 10)
            cells = split_polygon_into_voronoi(poly, num_points)
        elif method == 'bsp':
            depth = kwargs.get('depth', 3)
            cells = bsp_split(poly, depth)
        elif method == 'hex':
            hex_size = kwargs.get('hex_size', 20)
            cells = create_hex_grid(poly, hex_size)
        else:
            raise ValueError(f"Unknown subdivision method: {method}")

        # Save cells as list of dicts with id, coords, edge_ids
        block['cells'] = []
        for cell in cells:
            if isinstance(cell, Polygon):
                cell_coords = list(cell.exterior.coords)
            elif isinstance(cell, MultiPolygon):
                # If it's multipolygon, flatten it into separate cells
                for subcell in cell.geoms:
                    cell_coords = list(subcell.exterior.coords)
                    block['cells'].append({
                        'id': cell_counter,
                        'coords': cell_coords,
                        'edge_ids': []
                    })
                    print(f"Creating cell {cell_counter}")
                    cell_counter += 1
                continue  # Skip adding again below
            else:
                continue  # Unexpected type, skip

            block['cells'].append({
                'id': cell_counter,
                'coords': cell_coords,
                'edge_ids': []
            })
            cell_counter += 1


from shapely.geometry import Polygon, LineString

def associate_edges_to_cells(blocks):
    """For each block, associate block edge IDs to any cells touching the block boundary."""
    for block in blocks:
        coords = block['polygon']
        edge_ids = block.get('edge_ids', [])
        block_edges = []

        # Build LineStrings for each block edge
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            eid = edge_ids[i] if i < len(edge_ids) else None
            block_edges.append((LineString([p1, p2]), eid))

        new_cells = []
        for cell in block.get('cells', []):
            # Depending on your data, a cell might already be a dict or just coords
            if isinstance(cell, dict):
                cell_coords = cell['coords']
            else:
                cell_coords = cell

            cell_poly = Polygon(cell_coords)
            associated_edge_ids = []

            for edge_line, edge_id in block_edges:
                if edge_id is not None and cell_poly.touches(edge_line):
                    associated_edge_ids.append(edge_id)

            # Build new cell record
            new_cell = {
                'coords': cell_coords,
                'edge_ids': associated_edge_ids,
                'id': cell['id']
            }
            new_cells.append(new_cell)

        # Replace the cells with enriched cells
        block['cells'] = new_cells


def lcg_sequence(seed, max_val, min_val, length):
    """Linear Congruential Generator (LCG) sequence generator."""
    if max_val is None:
        max_val = 1
    if min_val is None:
        min_val = 0

    result = []
    for _ in range(int(length)):  # Ensure integer length
        seed = (seed * 9301 + 49297) % 233280
        rnd = seed / 233280
        result.append(round(min_val + rnd * (max_val - min_val)))
        seed += 1  # Increment seed

    return result

# Subdivide blocks

num_rows = lcg_sequence(39,1,15,len(blocks))
num_cols = lcg_sequence(27,1,15,len(blocks))
use_methods  = lcg_sequence(39,0,4,len(blocks))

subdivide_blocks(blocks, use_methods, hex_size=5, rows=num_rows, cols=num_cols)

associate_edges_to_cells(blocks)

filename = 'bezier_city_full.json'
with open(filename, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Saved updated blocks to {filename}!")


import random

# Helper to generate a random neon color
def random_neon_color():
    neon_colors = [
        '#39FF14',  # Neon Green
        '#FF073A',  # Neon Red
        '#0FF0FC',  # Neon Cyan
        '#F800FF',  # Neon Magenta
        '#FE019A',  # Neon Pink
        '#FC6C85',  # Neon Coral
        '#DFFF00',  # Neon Yellow
        '#FF5F1F',  # Neon Orange
        '#08F7FE',  # Neon Electric Blue
        '#B10DC9'   # Strong Purple
    ]
    return random.choice(neon_colors)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

for block in blocks:
    for cells in block.get('cells', []):
        cell_coords = cells['coords']
        patch = plt.Polygon(
            cell_coords, 
            facecolor=random_neon_color(),  # ← neon magic here
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(patch)

    for street in block.get('cells', []):
        if isinstance(street, list):  # Usual polygon
            patch = plt.Polygon(street, alpha=0.5, edgecolor='black')
            ax.add_patch(patch)
        elif isinstance(street, LineString):  # L-system street
            x, y = street.xy
            ax.plot(x, y, color='cyan', linewidth=1)

    # Draw the block boundary too
    poly = Polygon(block['polygon'])
    if isinstance(poly, Polygon):
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', linewidth=1.5)
    elif isinstance(poly, MultiPolygon):
        for single_poly in poly.geoms:
            x, y = single_poly.exterior.xy
            ax.plot(x, y, color='#08F7FE', linewidth=1.5)

# Dynamic limits
all_x = []
all_y = []
for block in blocks:
    for cells in block.get('cells', []):
        cell_coords = cells['coords']
        for x, y in cell_coords:
            all_x.append(x)
            all_y.append(y)

ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
ax.set_aspect('equal')

# Dark background for neon pop
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

plt.title('Subdivision of Blocks (Neon City)', color='white')
plt.grid(False)
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

