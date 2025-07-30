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

# Download GTFS data from https://www.opendata-oepnv.de
# Extract to ./data folder (e.g., ./data/vrr/ or ./data/delfi/)

# Initialize database from GTFS data
uv run python main.py init --data-dir data/vrr    # For VRR regional data
# or
uv run python main.py init --data-dir data/delfi  # For Germany-wide data
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

The tool uses GTFS (General Transit Feed Specification) data from German public transport providers. You can download the data from:

**OpenData ÖPNV**: https://www.opendata-oepnv.de

The `./data` folder should contain GTFS files such as:
- `stops.txt/csv` - Transit stops with coordinates
- `routes.txt/csv` - Transit routes (bus, tram, S-Bahn, etc.)
- `trips.txt/csv` - Individual trips
- `stop_times.txt/csv` - Scheduled stops with times
- `transfers.txt/csv` - Transfer connections
- `calendar.txt/csv` - Service schedules
- `calendar_dates.txt/csv` - Service exceptions

Example datasets:
- **VRR (Rhein-Ruhr)**: ~34k stops, 1.8k routes, 247k trips, 5.7M stop times
- **DELFI (Germany-wide)**: Much larger dataset covering all of Germany

## Technical Details

### Performance Optimizations

- **Smart filtering**: Only includes stops within reasonable reach (50km/h max transit speed)
- **Journey limiting**: Maximum 1 stop per minute (prevents unrealistic routes)
- **On-demand graphs**: Builds optimized network per query instead of global graph
- **Geocoding cache**: Avoids repeated address lookups

### Database-Driven Algorithm Steps

1. **Geocode** starting address using OpenStreetMap/Nominatim
2. **Find walkable stops** within 20-minute walk of origin
3. **Optimize origins** using line coverage analysis (reduces redundancy)
4. **Line following** using SQL queries to explore transit routes
5. **Time window filtering** to only consider relevant trips
6. **Transfer handling** with realistic transfer times
7. **Walking expansion** around reachable transit stops

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

### Algorithm

The tool uses a **database-driven line-following algorithm** that:
- Uses targeted SQL queries instead of building graphs in memory
- Finds comprehensive results: ~343 reachable stops from Dortmund TU in 30 minutes  
- **Fast computation**: ~17 seconds (3.3x faster than previous graph approach)
- **Memory efficient**: No graph construction needed
- **Time window optimization**: Only considers trips within the analysis window

```bash
# Query with database-driven algorithm (default and only option)
uv run python main.py query --address "Emil-Figge-Str. 42, Dortmund" --time 30
```

### Recent Improvements (2025-01)

- **✅ Simplified codebase**: Removed graph-based algorithm, database-driven is now the only option
- **✅ Database-driven line-following**: 3.3x faster than previous graph approach
- **✅ Time window optimization**: SQL queries only consider trips within analysis window
- **✅ Schedule-based routing**: Uses actual GTFS timetables instead of estimates
- **✅ Calendar validation**: Only includes services running on specific dates
- **✅ Circle Union Visualization**: Precise circular walking areas based on remaining time
- **✅ Human-Readable Line Names**: Shows passenger-friendly names (447, U43, S1)
- **✅ Enhanced Transit Stop Markers**: Interactive markers with travel times and lines
- **✅ Memory efficiency**: No graph construction overhead
- **✅ GitHub Pages Compatible**: Static HTML files ready for web deployment

### Current Limitations

- No real-time data integration (uses static GTFS schedules)
- Walking connections between stops limited by performance thresholds
- Requires manual download of GTFS data from providers

## Next Steps

- **Export options**: GeoJSON, KML for use in mapping applications
- **Multi-modal routing**: Combine walking, cycling, and transit
- **Spatial indexing**: R-tree optimization for walking connections

## Attribution

This project uses the following open-source libraries and data sources:

### Libraries
- **[Folium](https://github.com/python-visualization/folium)** - Interactive map visualization (BSD-3-Clause)
- **[GeoPandas](https://github.com/geopandas/geopandas)** - Geographic data processing (BSD-3-Clause)
- **[Shapely](https://github.com/shapely/shapely)** - Geometric analysis (BSD-3-Clause)
- **[Pandas](https://github.com/pandas-dev/pandas)** - Data analysis (BSD-3-Clause)
- **[Click](https://github.com/pallets/click)** - Command-line interface (BSD-3-Clause)
- **[Rich](https://github.com/Textualize/rich)** - Terminal formatting (MIT)
- **[AlphaShape](https://github.com/bellockk/alphashape)** - Concave hull generation (MIT)
- **[PyProj](https://github.com/pyproj4/pyproj)** - Cartographic projections (MIT)
- **[NetworkX](https://github.com/networkx/networkx)** - Graph algorithms (BSD-3-Clause)
- **[GeoPy](https://github.com/geopy/geopy)** - Geocoding (MIT)

### Data Sources
- **[OpenStreetMap](https://www.openstreetmap.org/copyright)** - Map tiles and geocoding data (© OpenStreetMap contributors, ODbL)
- **[VRR GTFS Data](https://www.opendata-oepnv.de)** - Public transport schedules for Rhine-Ruhr region
- **[HVV GTFS Data](https://www.opendata-oepnv.de)** - Public transport schedules for Hamburg region
- **[DELFI GTFS Data](https://www.opendata-oepnv.de)** - Germany-wide public transport schedules

### Map Visualization
Maps are generated using Folium with OpenStreetMap tiles. All interactive maps include proper attribution as required by the OpenStreetMap license.