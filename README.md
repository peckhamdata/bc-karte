# Die neue Karte von Bézier City

## 🧰 Requirements

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### 📦 Install dependencies

If you haven't already:

```bash
poetry install

## Generate the map with

```bash
poetry run python city_builder.py

This will:

* Generate Bézier and diagonal streets.
* Detect intersections and annotate junctions.
* Render the map as bezier_city.png.
* Save street geometry (with junctions) to bezier_city.json.

📁 Outputs

* bezier_city.png: Visualization of the generated map.
* bezier_city.json: JSON representation of the city structure.
