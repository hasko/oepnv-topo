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

# Future commands:
# - visualize: Generate isochrone maps and export
```

## Key Functions

- `geocode_address()` - OpenStreetMap geocoding with caching
- `build_transit_graph()` - Optimized graph building with distance/time filtering  
- `haversine_distance()` - Calculate distance between coordinates in km
- `find_nearest_stops()` - Spatial query to find stops near a location
- NetworkX Dijkstra's algorithm for isochrone calculation

## Implementation Status

✅ **Completed Core Features:**
1. ✅ Geocode German addresses using OpenStreetMap/Nominatim
2. ✅ Build optimized transit graphs from GTFS data
3. ✅ Calculate isochrones (reachable areas) within time limits
4. ✅ Smart filtering: distance-based (50km/h max) and journey length (1 stop/min)
5. ✅ On-demand graph building for efficient queries

**Performance Optimizations:**
- Filters 34k+ stops down to ~2-10k based on reachable distance
- Processes ~200k-1M connections instead of 5.7M total
- Uses Dijkstra's algorithm for accurate travel time calculations
- Caches geocoding results to avoid repeated API calls

**Next Steps:**
- Visualization: Generate isochrone maps using folium or matplotlib
- Export: Save results as GeoJSON or other map formats
- Time-dependent routing: Use actual departure times from schedules