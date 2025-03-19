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

# Initialize CityBuilder
seed = 1024
num_curves = 16
scale = 1

city_builder = CityBuilder(seed, num_curves, scale)

# Print values for verification
print("Seed:", city_builder.seed)
print("Magic:", city_builder.magic)
print("Bezier Sequence:", city_builder.bezier_sequence)
