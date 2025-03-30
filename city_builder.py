import numpy as np
from shapely.geometry import LineString, Point
from itertools import combinations
from scipy.spatial import KDTree


def qb(P0, P1, P2, num_points=100):
    """Generate points along a quadratic Bézier curve."""
    t_values = np.linspace(0, 1, num_points)
    bezier_points = np.column_stack([
        (1 - t_values) ** 2 * P0[0] + 2 * (1 - t_values) * t_values * P1[0] + t_values ** 2 * P2[0],
        (1 - t_values) ** 2 * P0[1] + 2 * (1 - t_values) * t_values * P1[1] + t_values ** 2 * P2[1],
    ])
    return bezier_points  # shape: (num_points, 2)

class CityBuilder:
    def __init__(self, seed, num_curves, scale):
        self.seed = seed
        self.scale = scale
        self.size = seed * scale
        self.magic = self.seed / 2
        self.num_curves = num_curves
        self.next_street = -1
        self.curve_num_points = self.seed
        self.verbose = False
        self.bezier_streets = []
        self.bresham_streets = []
        self.cols = []
        self.lines = []

        # Generate LCG sequence and sort it
        self.bezier_sequence = sorted(self.lcg_sequence(self.magic, self.magic, 0, self.magic)[:num_curves])


    def lcg_sequence(self, seed, max_val, min_val, length):
        """Linear Congruential Generator (LCG) sequence generator."""
        if max_val is None:
            max_val = 1
        if min_val is None:
            min_val = 0

        result = []
        for _ in range(int(length)):  # Ensure integer length
            seed = (seed * 9301 + 49297) % 233280
            rnd = seed / 233280
            result.append(min_val + rnd * (max_val - min_val))
            seed += 1  # Increment seed

        return result


    def circle(self, radius):
        """Generate a set of points forming a circle."""
        offset = (self.seed / 2) * self.scale
        points = []

        # Iterate over angles in radians from 0 to 7, step size π/360
        for angle in np.arange(0, 7, np.pi / 360):
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            points.append((x + offset, y + offset))  # Store as tuples

        return points

    def street_id(self):
        """Generate a unique street ID."""
        self.next_street += 1
        return self.next_street

    def quadratic_bezier(self, P0, P1, P2, num_points=100):
        """Generate points along a quadratic Bézier curve."""
        t_values = np.linspace(0, 1, num_points)
        bezier_points = [( (1-t)**2 * P0[0] + 2*(1-t)*t * P1[0] + t**2 * P2[0],
                           (1-t)**2 * P0[1] + 2*(1-t)*t * P1[1] + t**2 * P2[1] ) for t in t_values]
        return bezier_points


    def build_bezier_streets(self):
        """Generate Bézier streets using circular points."""
        circle_points = self.circle((self.seed / 2) * self.scale)
        inner_circle = self.circle((self.seed / 2 - 128) * self.scale)

        circle_offset = len(circle_points) // 8  # #MAGIC
        inner_circle_length = len(inner_circle)

        # Extend lists to prevent out-of-range errors
        circle_points *= 4  # #HACK
        inner_circle *= 4  # #HACK

        for i in range(self.num_curves):
            here = int(self.bezier_sequence[i]) + 50
            there = here + (inner_circle_length // 2)

            curve = {
                "start": {"x": circle_points[here][0], "y": circle_points[here][1]},
                "control": {"x": inner_circle[there][0], "y": inner_circle[there][1]},
                "end": {"x": circle_points[here + circle_offset][0], "y": circle_points[here + circle_offset][1]}
            }

            self.bezier_streets.append({
                "id": self.street_id(),
                "type": "bezier",
                "geometry": curve,
                "junctions": []
            })

        return self.bezier_streets

    def add_junctions(self):
        """Find intersections of Bézier curves and add junctions, including connected street IDs."""
        all_points = []
        for street in self.bezier_streets:
            P0 = (street["geometry"]["start"]["x"], street["geometry"]["start"]["y"])
            P1 = (street["geometry"]["control"]["x"], street["geometry"]["control"]["y"])
            P2 = (street["geometry"]["end"]["x"], street["geometry"]["end"]["y"])
            curve_points = self.quadratic_bezier(P0, P1, P2, num_points=50)
            all_points.append((street["id"], curve_points))

        # Compare each pair of streets for intersections
        for (id1, points1), (id2, points2) in combinations(all_points, 2):
            for p1, p2 in zip(points1[:-1], points1[1:]):
                for q1, q2 in zip(points2[:-1], points2[1:]):
                    intersection = self.line_intersection(p1, p2, q1, q2)
                    if intersection:
                        x, y = intersection
                        # Add to id1
                        for street in self.bezier_streets:
                            if street["id"] == id1:
                                street.setdefault("junctions", []).append({
                                    "x": x,
                                    "y": y,
                                    "street_id": id2
                                })
                            elif street["id"] == id2:
                                street.setdefault("junctions", []).append({
                                    "x": x,
                                    "y": y,
                                    "street_id": id1
                                })


    def line_intersection(self, p1, p2, q1, q2):
        """Find intersection of two line segments if they intersect."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Parallel lines
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))  # Intersection point
        return None

###############################################################################

    def build_diagonal_streets(self):
        self.diagonal_streets = []
        self.cols = []
        self.streets = self.bezier_streets.copy()  # Start with the existing Bézier streets

        for i in range(1, self.num_curves):
            lines = []
            # #MAGIC: j ranges from 0 to curve_num_points - 100 in steps of 50
            for j in range(0, self.curve_num_points - 100, 50):
                k = j
                offset = 100  # static for now, as per original JS

                # Generate points along both Bézier curves
                curr = self.bezier_streets[i]["geometry"]
                prev = self.bezier_streets[i - 1]["geometry"]

                curve_points = self.quadratic_bezier(
                    (curr["start"]["x"], curr["start"]["y"]),
                    (curr["control"]["x"], curr["control"]["y"]),
                    (curr["end"]["x"], curr["end"]["y"]),
                    num_points=self.curve_num_points
                )

                prev_curve_points = self.quadratic_bezier(
                    (prev["start"]["x"], prev["start"]["y"]),
                    (prev["control"]["x"], prev["control"]["y"]),
                    (prev["end"]["x"], prev["end"]["y"]),
                    num_points=self.curve_num_points
                )

                # Safeguard: avoid going out of bounds
                if k + offset >= len(prev_curve_points):
                    continue

                street = {
                    "id": self.street_id(),
                    "type": "bresenham",
                    "junctions": [{"x": curve_points[k][0], "y": curve_points[k][1], "street_id": self.bezier_streets[i]["id"]},
                                  {"x": prev_curve_points[k + offset][0], "y": prev_curve_points[k + offset][1], "street_id": self.bezier_streets[i - 1]["id"]}],
                    "geometry": {
                        "start": {"x": curve_points[k][0], "y": curve_points[k][1]},
                        "end": {"x": prev_curve_points[k + offset][0], "y": prev_curve_points[k + offset][1]}
                    }
                }
                self.streets.append(street)
                self.diagonal_streets.append(street)
                lines.append(street)

            self.cols.append(lines)


    def add_bresenham_junctions(self):
        """
        Find intersections between bresenham-type streets and record junctions.
        
        Adds a {"x", "y", "street_id"} dict to each bresenham street's 'junctions' list when it intersects another.
        """

        # Filter for bresenham streets only
        bresenham_streets = [s for s in self.streets if s.get("type") == "bresenham"]

        for i, from_street in enumerate(bresenham_streets):
            for j, to_street in enumerate(bresenham_streets):
                if from_street["id"] == to_street["id"]:
                    continue  # skip self

                # Extract segment endpoints
                x1, y1 = from_street["geometry"]["start"]["x"], from_street["geometry"]["start"]["y"]
                x2, y2 = from_street["geometry"]["end"]["x"], from_street["geometry"]["end"]["y"]
                x3, y3 = to_street["geometry"]["start"]["x"], to_street["geometry"]["start"]["y"]
                x4, y4 = to_street["geometry"]["end"]["x"], to_street["geometry"]["end"]["y"]

                intersection = self.line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                if intersection:
                    x, y = intersection
                    from_street.setdefault("junctions", []).append({
                        "x": x,
                        "y": y,
                        "street_id": to_street["id"]
                    })

###############################################################################

    def sync_bresenham_junctions_to_beziers(self):
        """
        Ensure all Bézier streets are aware of junctions with Bresenham (diagonal) streets.
        """
        # Build a quick lookup of streets by ID
        street_lookup = {street["id"]: street for street in self.bezier_streets + self.streets}

        for street in self.streets:
            if street["type"] != "bresenham":
                continue  # Only care about diagonal connections here

            for junction in street.get("junctions", []):
                target_id = junction["street_id"]
                if target_id not in street_lookup:
                    continue

                target_street = street_lookup[target_id]
                # Check if this junction already exists in the target
                exists = any(
                    abs(j["x"] - junction["x"]) < 1e-6 and
                    abs(j["y"] - junction["y"]) < 1e-6 and
                    j.get("street_id") == street["id"]
                    for j in target_street.get("junctions", [])
                )

                if not exists:
                    if "junctions" not in target_street:
                        target_street["junctions"] = []
                    target_street["junctions"].append({
                        "x": junction["x"],
                        "y": junction["y"],
                        "street_id": street["id"]
                    })

###############################################################################

    def sort_junctions_along_streets(self):
        """Sort junctions along each street from start to end based on Euclidean distance."""
        for street in self.streets:
            if street["type"] != "bezier":
                start_x = street["geometry"]["start"]["x"]
                start_y = street["geometry"]["start"]["y"]

                def distance(junction):
                    dx = junction["x"] - start_x
                    dy = junction["y"] - start_y
                    return dx * dx + dy * dy  # no need for sqrt, sorting only

                street["junctions"].sort(key=distance)


    def sort_junctions_along_bezier(self):

        for street in self.streets:
            if street["type"] == "bezier":
                P0 = np.array([street["geometry"]["start"]["x"], street["geometry"]["start"]["y"]])
                P1 = np.array([street["geometry"]["control"]["x"], street["geometry"]["control"]["y"]])
                P2 = np.array([street["geometry"]["end"]["x"], street["geometry"]["end"]["y"]])

                # Generate sampled points along the curve
                bezier_points = np.array(qb(P0, P1, P2,))
                # Build a KDTree for fast nearest-neighbor lookup
                try:
                    tree = KDTree(bezier_points)
                except ValueError as e:
                    print(f"KDTree error: {e}")
                    continue

                # Find closest index for each junction
                indexed_junctions = []
                for j in street["junctions"]:
                    try:
                        point = np.array([j['x'], j['y']]).reshape(1, -1)
                        dist, idx = tree.query(point)
                        indexed_junctions.append((j, idx[0]))
                    except Exception as e:
                        print(f"Error querying KDTree: {e}")
                        print(j)
                        print(tree)
                        import pdb; pdb.set_trace()
                # Sort junctions by their index along the curve
                sorted_junctions = [j for j, _ in sorted(indexed_junctions, key=lambda tup: tup[1])]
                street["junctions"] = sorted_junctions


###############################################################################

import numpy as np
import matplotlib.pyplot as plt

def quadratic_bezier(P0, P1, P2, num_points=50):
    """
    Generates a quadratic Bézier curve given three control points.

    :param P0: Start point (x, y)
    :param P1: Control point (x, y)
    :param P2: End point (x, y)
    :param num_points: Number of points to interpolate along the curve.
    :return: Two lists containing x and y coordinates of the curve.
    """
    t_values = np.linspace(0, 1, num_points)
    bezier_x = (1 - t_values) ** 2 * P0[0] + 2 * (1 - t_values) * t_values * P1[0] + t_values ** 2 * P2[0]
    bezier_y = (1 - t_values) ** 2 * P0[1] + 2 * (1 - t_values) * t_values * P1[1] + t_values ** 2 * P2[1]
    return bezier_x, bezier_y



def plot_city_from_dict(city_data, output_png="bezier_city.png"):
    """
    Plots the Bézier and diagonal city streets from a dictionary, including junctions, and saves it as a PNG.

    :param city_data: Dictionary containing street data.
    :param output_png: Filename for saving the PNG output.
    """

    plt.figure(figsize=(10, 10))

    for street in city_data["streets"]:
        geometry = street["geometry"]
        street_type = street.get("type", "bezier")
        if street_type == "bezier":
            # Extract Bézier control points
            P0 = (geometry["start"]["x"], geometry["start"]["y"])
            P1 = (geometry["control"]["x"], geometry["control"]["y"])
            P2 = (geometry["end"]["x"], geometry["end"]["y"])

            bezier_x, bezier_y = quadratic_bezier(P0, P1, P2, num_points=100)
            plt.plot(bezier_x, bezier_y, 'black', linewidth=1)

        elif street_type == "bresenham":
            # Draw straight lines for diagonal streets
            x = [geometry["start"]["x"], geometry["end"]["x"]]
            y = [geometry["start"]["y"], geometry["end"]["y"]]
            plt.plot(x, y, 'blue', linestyle='--', linewidth=0.8)
 
        # Plot junctions
        if "junctions" in street:
            for junction in street["junctions"]:
                plt.scatter(junction["x"], junction["y"], color='red', s=20, edgecolors='black', zorder=3)


    plt.title("Bézier City Map with Diagonal Streets and Junctions")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(output_png, dpi=300)
    plt.show()






# Example usage:
city_builder = CityBuilder(seed=1024, num_curves=16, scale=1)  # Create city builder instance
city_builder.build_bezier_streets()
city_builder.build_diagonal_streets()
city_builder.add_junctions()
city_builder.add_bresenham_junctions()
city_builder.sync_bresenham_junctions_to_beziers()
city_builder.sort_junctions_along_streets()
city_builder.sort_junctions_along_bezier()

city_data = {"streets": city_builder.streets}

from json import dumps
open ("bezier_city.json", "w").write(dumps(city_data))
# plot_city_from_dict(city_data, output_png="bezier_city.png")