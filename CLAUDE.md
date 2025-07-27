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

# Future commands will include:
# - query: Find stops within travel time from an address
# - visualize: Generate topological map
```

## Key Functions

- `haversine_distance()` - Calculate distance between coordinates in km
- `find_nearest_stops()` - Spatial query to find stops near a location
- Database schema includes proper indexes for efficient spatial queries

## Implementation Notes

The main algorithm should:
1. Geocode the input German address to coordinates
2. Build a graph of the transit network from GTFS data
3. Calculate travel times from the origin to all reachable stops
4. Generate a topological visualization showing travel time zones

The GTFS stop_times.csv file is very large (~180MB, 5.7M records) - the import uses pandas chunking for efficient processing.