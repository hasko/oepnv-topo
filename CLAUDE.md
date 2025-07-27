# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

oepnv-topo creates topological maps based on public transport travel durations in Germany. It calculates and visualizes areas reachable within a specified time from a given address using VRR (Verkehrsverbund Rhein-Ruhr) public transport data.

## Key Architecture

The project uses GTFS (General Transit Feed Specification) format data stored in `/data/`:
- `stops.csv` - Transit stop locations with coordinates (stop_id, stop_name, stop_lat, stop_lon)
- `routes.csv` - Transit routes with types (route_id, route_short_name, route_type)
- `trips.csv` - Individual trips on routes
- `stop_times.csv` - Schedule information (large file ~180MB)
- `transfers.csv` - Transfer times between stops
- `linien.csv` and `haltestellen.csv` - Additional German transit data

All data is imported into a SQLite database (`oepnv.db`) for efficient querying.

## Development Setup

The project uses Python 3.11 with uv for dependency management. Key dependencies:
- `click` - Command-line interface framework
- `rich` - Terminal formatting and progress bars
- `pandas` - Efficient CSV processing for large files
- `folium` - Interactive map generation with OpenStreetMap tiles
- `geopandas` - Geographic data processing and spatial operations
- `alphashape` - Creating realistic polygon boundaries from point data

## Commands

```bash
# Initialize database from GTFS data
uv run python main.py init

# Show database statistics and example queries
uv run python main.py stats

# Find reachable stops within travel time (isochrone calculation)
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 30
uv run python main.py query --lat 51.2197 --lon 6.7943 --time 20

# Build optimized graph for specific location (optional)
uv run python main.py build-graph --lat 51.2197 --lon 6.7943 --time 30

# Generate interactive maps with isochrone visualization
uv run python main.py visualize --address "Düsseldorf Hauptbahnhof" --time 30
uv run python main.py visualize --lat 51.2197 --lon 6.7943 --time 20 --simple

# Combine query with automatic map generation
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 20 --visualize
```

## Key Functions

- `geocode_address()` - OpenStreetMap geocoding with caching
- `build_transit_graph()` - Optimized graph building with distance/time filtering  
- `find_walkable_stops()` - Find all stops within 20-minute walk
- `optimize_walkable_stops_by_line_coverage()` - Line coverage optimization
- `calculate_multi_origin_isochrone()` - Multi-origin Dijkstra algorithm
- `add_end_walking_expansion()` - 20-minute walking from transit destinations
- `haversine_distance()` - Calculate distance between coordinates in km
- `create_isochrone_map()` - Generate time-layered interactive maps with Folium
- `create_simple_boundary_map()` - Generate single polygon boundary maps
- `points_to_polygon()` - Convert coordinate points to alpha shape polygons

## Implementation Status

✅ **Completed Core Features:**
1. ✅ Geocode German addresses using OpenStreetMap/Nominatim
2. ✅ Build optimized transit graphs from GTFS data
3. ✅ Enhanced walking model: 20 minutes at start and end of journey
4. ✅ Line coverage optimization: Smart origin selection (10x speedup)
5. ✅ Multi-origin isochrone calculation with time budgeting
6. ✅ End-of-journey walking expansion to destinations
7. ✅ Interactive map visualization with OpenStreetMap tiles
8. ✅ Time-layered polygon overlays with realistic boundaries

**Enhanced Walking Model:**
- Multi-origin Dijkstra from all stops within 20-minute walk
- Line coverage optimization: 320 → 30 origins while covering all 80 lines
- End walking expansion: 20-minute radius from transit destinations
- Results: 3,173 reachable points vs ~5 with basic model

**Interactive Visualization Features:**
- Folium-based interactive maps with OpenStreetMap base tiles
- Alpha shape polygons: Realistic concave boundaries handling coverage gaps
- Time-layered overlays: Color-coded zones (0-10min, 10-20min, 20-30min, etc.)
- Transit stop markers: Individual stops with travel time popups
- Professional styling: Legend, tooltips, zoom/pan functionality
- CLI integration: Standalone and integrated with query command
- Optimized colors: Magenta/purple scheme avoids conflicts with map green areas

**Performance Optimizations:**
- Filters 34k+ stops down to ~2-10k based on reachable distance  
- Line coverage reduces Dijkstra origins by 10.7x
- Processes ~200k-1M connections instead of 5.7M total
- Caches geocoding results to avoid repeated API calls

**Next Steps:**
- Export: Save results as GeoJSON or other map formats
- Spatial indexing: R-tree optimization for walking connections
- Time-dependent routing: Use actual departure times from schedules