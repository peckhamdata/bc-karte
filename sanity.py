import json

# Load the city data
filename = "bezier_city_full.json"

with open(filename) as f:
    data = json.load(f)

blocks = data['blocks']

# Set how many you want to print
N = 5

# --- Print first N blocks ---
print(f"\nFirst {N} Blocks:\n" + "-" * 30)
for block in blocks[:N]:
    print(f"Block ID: {block.get('id')}")
    print(f"Polygon Points: {len(block.get('polygon', []))} points")
    print(f"Edge IDs: {block.get('edge_ids')}")
    print(f"Street IDs: {block.get('street_ids')}")
    if 'cells' in block and block['cells']:
        first_cell = block['cells'][0]
        if isinstance(first_cell, dict):
            print(f"First Cell ID: {first_cell.get('id', 'No ID')} Edge IDs: {first_cell.get('edge_ids', [])}")
        else:
            print(f"First Cell: {first_cell[:3]}... (no ID)")  # Just show part of coords
    print("-" * 30)

# --- Build streets from block polygons ---
streets = []
for block in blocks:
    coords = block['polygon']
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        streets.append((p1, p2))

# --- Print first N streets ---
print(f"\nFirst {N} Streets:\n" + "-" * 30)
for idx, (start, end) in enumerate(streets[:N]):
    print(f"Street {idx}: {start} -> {end}")
print("-" * 30)

