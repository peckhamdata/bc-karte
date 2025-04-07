from pydantic import BaseModel
from typing import List, Optional
from shapely.geometry import Polygon, LineString




class Edge(BaseModel):
    id: int
    geometry: List[List[float]]  # two points: start and end
    street_id: int
    junction_ids: List[int]


class Street(BaseModel):
    id: int
    edge_ids: List[int]

class Cell(BaseModel):
    id: int
    coords: List[List[float]]
    edge_ids: List[int] = []

class Block(BaseModel):
    id: int
    polygon: List[List[float]]
    edge_ids: List[int]
    street_ids: List[int]
    cells: List[Cell] = []

class CityModel(BaseModel):
    blocks: List[Block]
    streets: List[Street]
    edges: List[Edge]


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Bezier City API")

# Allow browser frontends if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your city once at startup
with open('bezier_city_full.json') as f:
    city_data = json.load(f)


city = CityModel(**city_data)

print("\n=== City Debug Info ===")
print(f"Total blocks: {len(city.blocks)}")
print(f"Total streets: {len(city.streets)}")
print(f"Total edges: {len(city.edges)}")
total_cells = sum(len(block.cells) for block in city.blocks)
print(f"Total cells: {total_cells}")

for block in city.blocks[:3]:
    print(f"\nBlock ID: {block.id}")
    print(f"  Polygon points: {len(block.polygon)}")
    print(f"  Edge IDs: {block.edge_ids[:5]}{'...' if len(block.edge_ids) > 5 else ''}")
    print(f"  Street IDs: {block.street_ids[:5]}{'...' if len(block.street_ids) > 5 else ''}")
    if block.cells:
        print(f"  First Cell ID: {block.cells[0].id}, Edge IDs: {block.cells[0].edge_ids}")

for street in city.streets[:3]:
    print(f"\nStreet ID: {street.id}")
    print(f"  Edge IDs: {street.edge_ids[:5]}{'...' if len(street.edge_ids) > 5 else ''}")

for edge in city.edges[:3]:
    print(f"\nEdge ID: {edge.id}")
    print(f"  Geometry: {edge.geometry}")
    print(f"  Street ID: {edge.street_id}")
    print(f"  Junction IDs: {edge.junction_ids}")

print("\n=== Debug Complete ===")



@app.get("/blocks", response_model=List[Block])
async def get_blocks():
    return city.blocks

@app.get("/blocks/{block_id}", response_model=Block)
async def get_block(block_id: int):
    for block in city.blocks:
        if block.id == block_id:
            return block
    raise HTTPException(status_code=404, detail="Block not found")

@app.get("/streets", response_model=List[Street])
async def get_streets():
    return city.streets

@app.get("/streets/{street_id}", response_model=Street)
async def get_street(street_id: int):
    for street in city.streets:
        if street.id == street_id:
            return street
    raise HTTPException(status_code=404, detail="Street not found")

@app.get("/edges", response_model=List[Edge])
async def get_edges():
    return city.edges

@app.get("/edges/{edge_id}", response_model=Edge)
async def get_edge(edge_id: int):
    for edge in city.edges:
        if edge.id == edge_id:
            return edge
    raise HTTPException(status_code=404, detail="Edge not found")

@app.get("/cells", response_model=List[Cell])
async def get_cells():
    cells = []
    for block in city.blocks:
        cells.extend(block.cells)
    return cells

@app.get("/cells/{cell_id}", response_model=Cell)
async def get_cell(cell_id: int):
    for block in city.blocks:
        for cell in block.cells:
            if cell.id == cell_id:
                return cell
    raise HTTPException(status_code=404, detail="Cell not found")
