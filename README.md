# ÖPNV Topo - Public Transport Isochrone Calculator

Creates isochrone maps showing reachable areas via public transport from any German address.

## Description

This tool calculates how far you can travel using public transport (buses, trains, trams) within a given time limit from any starting address in Germany. It uses real GTFS (General Transit Feed Specification) data from VRR (Verkehrsverbund Rhein-Ruhr) covering the Rhine-Ruhr metropolitan area.

## Features

- **Address geocoding** using OpenStreetMap/Nominatim
- **Isochrone calculation** - find all stops reachable within time limit
- **Interactive map visualization** - OpenStreetMap-based maps with transparent overlays
- **Optimized performance** - smart filtering reduces 34k+ stops to relevant subset
- **Realistic modeling** - accounts for transfers, waiting times, and walking
- **Multiple input methods** - address or coordinates
- **Rich terminal output** - progress bars and formatted results

## Example Results

From Düsseldorf Hauptbahnhof in 20 minutes with enhanced walking model:
- **3,173 total reachable points** (vs ~5 with basic model)
- **125 transit stops** reachable by public transport + walking
- **3,048 additional walking destinations** within 20min of transit stops
- **10.7x performance optimization** using line coverage (320 → 30 origins)

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

# Generate interactive map with precise circle union boundaries
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 30 --visualize

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

### Interactive Map Visualization

```bash
# Generate standalone interactive map
uv run python main.py visualize --address "Düsseldorf Hauptbahnhof" --time 30

# Create simple boundary map (single overlay)
uv run python main.py visualize --lat 51.2197 --lon 6.7943 --time 20 --simple

# Combine query calculation with automatic map generation
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 20 --visualize

# Custom output filename  
uv run python main.py visualize --address "Essen Hauptbahnhof" --time 30 --output essen_map.html

# Simple boundary map (faster generation)
uv run python main.py visualize --address "Köln Hauptbahnhof" --time 25 --simple
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

### Enhanced Walking Model

- **Start walking**: Up to 20 minutes to reach any transit stop (finds optimal origins)
- **End walking**: 20-minute radius from each transit-reachable stop
- **Line coverage optimization**: Smart selection of origins to avoid redundancy
- **Transfer walking**: 500m limit between stops (realistic transfer distance)

### Interactive Visualization Features

- **Circle Union Boundaries**: Precise walking areas computed as unions of circles around transit stops
- **Time-Based Layering**: Color-coded zones showing 0-10min, 10-20min, 20-30min travel times
- **Accurate Geometry**: Preserves holes, disconnected areas, and complex shapes that alpha shapes would approximate
- **Interactive Transit Markers**: Click/hover stops to see travel times and serving lines (447, U43, S1, etc.)
- **OpenStreetMap Integration**: Professional cartographic styling with free base tiles
- **Static HTML Output**: Self-contained files ready for GitHub Pages or web deployment
- **Responsive Design**: Works seamlessly on desktop and mobile browsers
- **Human-Readable Information**: Passenger-friendly line names and clear time indicators

### Recent Improvements (2025-01)

- **✅ Circle Union Visualization**: Replaced alpha shapes with precise circular walking areas based on remaining time budget
- **✅ Human-Readable Line Names**: Hover tooltips show passenger-friendly names (447, U43, S1) instead of technical IDs
- **✅ Enhanced Transit Stop Markers**: Interactive markers with travel times and serving line information
- **✅ Fixed walking time bug**: End walking time no longer incorrectly added to total duration
- **✅ Enhanced route connectivity**: Route 447 and other bus lines now properly connect destinations
- **✅ Eliminated wait times for same-route connections**: No artificial delays when staying on same vehicle
- **✅ Added transfer penalties**: 5-minute penalty only when changing between different routes
- **✅ Direction filtering**: Prevents inefficient U-turns, reduces graph size by ~46%
- **✅ Detailed progress reporting**: Shows connection processing, filtering, and graph statistics
- **✅ GitHub Pages Compatible**: Generated maps are static HTML files ready for web deployment

### Current Limitations

- Uses simplified wait times rather than actual schedule-based routing
- No real-time data integration
- Walking connections between stops limited by 2000-stop performance threshold

## Next Steps

- **Export options**: GeoJSON, KML for use in mapping applications
- **Time-dependent routing**: Use actual departure times from schedules
- **Spatial indexing**: R-tree optimization for walking connections