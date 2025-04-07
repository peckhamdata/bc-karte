import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import euclidean
from json import loads

# Define the quadratic Bezier function
def bezier_quad(P0, P1, P2, t):
    return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

# Function to find closest t index for a point
def closest_t(pt):
    return np.argmin([euclidean(pt, cp) for cp in curve_points]) / len(ts)


# Load your street data
with open("bezier_city.json", "r") as f:
    city_data = loads(f.read())

streets = city_data["streets"]
for street in streets:
    if street["type"] == "bezier":

        # Extract geometry
        P0 = np.array([street["geometry"]["start"]["x"], street["geometry"]["start"]["y"]])
        P1 = np.array([street["geometry"]["control"]["x"], street["geometry"]["control"]["y"]])
        P2 = np.array([street["geometry"]["end"]["x"], street["geometry"]["end"]["y"]])

        # Sample the curve at many t values
        ts = np.linspace(0, 1, 100)
        curve_points = [bezier_quad(P0, P1, P2, t) for t in ts]

        # Check each junction
        junctions = street["junctions"]

        # Compute t values
        junction_ts = [(j, closest_t((j['x'], j['y']))) for j in junctions]

        # # Print results
        # for j, t in junction_ts:
        #     print(f"Junction at ({j['x']:.1f}, {j['y']:.1f}) -> t ≈ {t:.3f}")

        # Optional: check if sorted
        sorted_ts = sorted(junction_ts, key=lambda jt: jt[1])
        is_sorted = all(j1[1] <= j2[1] for j1, j2 in zip(junction_ts, junction_ts[1:]))
        print(f"\nJunctions are {'already' if is_sorted else 'not'} sorted along the curve.")

