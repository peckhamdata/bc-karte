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

# Function to apply subdivision
def subdivide_blocks(blocks, method='rectangles', **kwargs):
    for idx, block in enumerate(blocks):
        coords = block['polygon']
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        if method == 'rectangles':
            rows = kwargs.get('rows')[idx]
            cols = kwargs.get('cols')[idx]
            cells = split_polygon_into_rectangles(poly, rows, cols)
        elif method == 'voronoi':
            num_points = kwargs.get('num_points', 10)
            cells = split_polygon_into_voronoi(poly, num_points)
        else:
            raise ValueError(f"Unknown subdivision method: {method}")

        # Save cells as lists of points
        block['cells'] = []
        for cell in cells:
            if isinstance(cell, Polygon):
                block['cells'].append(list(cell.exterior.coords))
            elif isinstance(cell, MultiPolygon):
                for subcell in cell.geoms:
                    block['cells'].append(list(subcell.exterior.coords))

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

import pdb; pdb.set_trace()
num_rows = lcg_sequence(39,1,15,len(blocks))
num_cols = lcg_sequence(27,1,15,len(blocks))

subdivide_blocks(blocks, method='rectangles', rows=num_rows, cols=num_cols)  # Change to 'voronoi' if you want

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
    for cell_coords in block.get('cells', []):
        patch = plt.Polygon(
            cell_coords, 
            facecolor=random_neon_color(),  # ← neon magic here
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(patch)
    
    # Draw the block boundary too
    poly = Polygon(block['polygon'])
    if isinstance(poly, Polygon):
        x, y = poly.exterior.xy
        ax.plot(x, y, color='white', linewidth=1.5)
    elif isinstance(poly, MultiPolygon):
        for single_poly in poly.geoms:
            x, y = single_poly.exterior.xy
            ax.plot(x, y, color='white', linewidth=1.5)

# Dynamic limits
all_x = []
all_y = []
for block in blocks:
    for cell_coords in block.get('cells', []):
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

