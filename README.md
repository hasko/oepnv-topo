# ÖPNV Topo - Public Transport Isochrone Calculator

Creates isochrone maps showing reachable areas via public transport from any German address.

## Description

This tool calculates how far you can travel using public transport (buses, trains, trams) within a given time limit from any starting address in Germany. It uses real GTFS (General Transit Feed Specification) data from VRR (Verkehrsverbund Rhein-Ruhr) covering the Rhine-Ruhr metropolitan area.

## Features

- **Address geocoding** using OpenStreetMap/Nominatim
- **Isochrone calculation** - find all stops reachable within time limit
- **Optimized performance** - smart filtering reduces 34k+ stops to relevant subset
- **Realistic modeling** - accounts for transfers, waiting times, and walking
- **Multiple input methods** - address or coordinates
- **Rich terminal output** - progress bars and formatted results

## Example Results

From Düsseldorf Hauptbahnhof in 30 minutes:
- 41 reachable stops total
- 11 stops within 15 minutes (local S-Bahn network)
- 30 stops within 15-30 minutes (regional connections to Neuss, Airport, etc.)

## Installation

Requires Python 3.11+ and uv for dependency management:

```bash
# Clone repository and install dependencies
uv sync

# Initialize database from GTFS data (requires ./data folder with VRR files)
uv run python main.py init
```

## Usage

### Basic Isochrone Query

```bash
# Find stops reachable within 30 minutes from an address
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 30

# Using coordinates instead of address
uv run python main.py query --lat 51.2197 --lon 6.7943 --time 20

# Shorter time limits for local area analysis
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 10
```

### Database Management

```bash
# Show database statistics
uv run python main.py stats

# Build optimized graph for specific location (optional)
uv run python main.py build-graph --lat 51.2197 --lon 6.7943 --time 30
```

## Data

The `./data` folder contains VRR (Verkehrsverbund Rhein-Ruhr) GTFS files:
- `stops.csv` - 34k+ transit stops with coordinates
- `routes.csv` - 1.8k+ transit routes (bus, tram, S-Bahn, etc.)
- `trips.csv` - 247k+ individual trips
- `stop_times.csv` - 5.7M+ scheduled stops with times
- `transfers.csv` - 78k+ transfer connections
- Additional German-specific files (`haltestellen.csv`, `linien.csv`)

## Technical Details

### Performance Optimizations

- **Smart filtering**: Only includes stops within reasonable reach (50km/h max transit speed)
- **Journey limiting**: Maximum 1 stop per minute (prevents unrealistic routes)
- **On-demand graphs**: Builds optimized network per query instead of global graph
- **Geocoding cache**: Avoids repeated address lookups

### Algorithm

1. Geocode starting address using OpenStreetMap/Nominatim
2. Filter stops to those within reachable distance 
3. Build directed graph with transit connections and transfer times
4. Run Dijkstra's algorithm to find shortest paths within time limit
5. Group and display results by travel time ranges

### Current Limitations

- Walking is currently limited to 500m transfers between stops
- Uses simplified wait times rather than actual schedule-based routing
- No real-time data integration

## Next Steps

- **Enhanced walking model**: 20-minute walking at start/end of journey
- **Visualization**: Generate interactive maps showing isochrone areas
- **Export options**: GeoJSON, KML for use in mapping applications
- **Time-dependent routing**: Use actual departure times from schedules