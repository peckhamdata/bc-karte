import numpy as np

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
    Plots the Bézier city streets from a dictionary and saves it as a PNG.

    :param city_data: Dictionary containing street data.
    :param output_png: Filename for saving the PNG output.
    """

    plt.figure(figsize=(10, 10))

    # Plot each Bézier street
    for street in city_data["streets"]:
        geometry = street["geometry"]  # This is a dictionary with start, control, end points

        # Extract start, control, and end points
        P0 = (geometry["start"]["x"], geometry["start"]["y"])
        P1 = (geometry["control"]["x"], geometry["control"]["y"])
        P2 = (geometry["end"]["x"], geometry["end"]["y"])

        # Generate Bézier curve points
        bezier_x, bezier_y = quadratic_bezier(P0, P1, P2, num_points=100)

        # Plot the Bézier curve
        plt.plot(bezier_x, bezier_y, linewidth=1, label=f"Street {street['id']}")

    plt.title("Bézier City Map")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend(loc="upper right", fontsize="small", ncol=2, frameon=False)
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Save and show the plot
    plt.savefig(output_png, dpi=300)
    plt.show()



# Example usage:
city_builder = CityBuilder(seed=1024, num_curves=16, scale=1)  # Create city builder instance
city_data = {"streets": city_builder.build_bezier_streets()}  # Generate Bézier streets

plot_city_from_dict(city_data)  # Plot from the dictionary
