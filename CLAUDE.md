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

## Available Datasets

Two datasets are available in the `data/` folder:

### VRR Dataset (`data/vrr/`)
- **Coverage**: VRR region (Ruhr area) 
- **Size**: 602MB, ~6.7M records
- **Files**: `.csv` format with German-specific `haltestellen.csv`, `linien.csv`
- **Performance**: Fast, 374 stops reachable in 30min from Emil-Figge-Str

### DELFI Dataset (`data/delfi/`)
- **Coverage**: All of Germany (~1,167 agencies)
- **Size**: 3.4GB, ~50.6M records  
- **Files**: `.txt` format (GTFS standard)
- **Period**: July 5 - December 13, 2025
- **Performance**: Optimized with strategic database indexes

### HVV Dataset (`data/hvv/`)
- **Coverage**: Hamburg region
- **Size**: Medium-scale regional data
- **Files**: GTFS format with mixed time formats (8:00:00 vs 08:00:00)
- **Performance**: Fast with robust time format handling

## Commands

```bash
# Initialize database from specific dataset folder
uv run python main.py init --data-dir data/vrr     # VRR regional data
uv run python main.py init --data-dir data/delfi   # Germany-wide data

# Show database statistics and example queries
uv run python main.py stats

# Find reachable stops using database-driven algorithm
# VRR dataset: use dates in April 2025
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 30 --date 20250407
uv run python main.py query --lat 51.2197 --lon 6.7943 --time 20 --date 20250407

# DELFI dataset: use dates July-December 2025
uv run python main.py query --address "Emil-Figge-Str. 42, Dortmund" --time 30 --date 20250728

# HVV dataset: use dates in January 2025
uv run python main.py query --db-path data/hvv.db --address "Hamburg Hauptbahnhof" --time 20 --date 20250114

# Auto-select optimal date (shows prominent display)
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 30

# Specify custom departure time  
uv run python main.py query --address "Emil-Figge-Str. 42, Dortmund" --time 30 --date 20250728 --departure 08:00

# Generate interactive maps with reference date/time display
uv run python main.py query --address "Düsseldorf Hauptbahnhof" --time 30 --visualize
uv run python main.py query --lat 51.2197 --lon 6.7943 --time 20 --visualize --map-output custom_map.html

# Visualize existing results
uv run python main.py visualize --address "Düsseldorf Hauptbahnhof" --time 30
```

## Key Functions

- `geocode_address()` - OpenStreetMap geocoding with caching
- `is_service_active_on_date()` - Check if GTFS service runs on specific date using calendar data
- `get_active_trips_on_date()` - Get all trips running on a specific date
- `build_time_dependent_graph()` - Build time-expanded graph with actual departure/arrival times
- `database_line_following_isochrone()` - Database-driven algorithm using SQL queries (3.3x faster)
- `find_walkable_stops()` - Find all stops within 20-minute walk
- `optimize_walkable_stops_by_line_coverage()` - Line coverage optimization
- `calculate_time_dependent_isochrone()` - Schedule-based Dijkstra with actual departure times
- `add_schedule_based_walking_expansion()` - Walking expansion with circle unions for schedule-based results
- `haversine_distance()` - Calculate distance between coordinates in km
- `create_circle_union_map()` - Generate interactive maps with precise circular boundaries and line information
- `get_stop_lines_mapping()` - Extract human-readable line names (447, U43, S1) from GTFS data

## Implementation Status

✅ **Completed Core Features:**
1. ✅ Schedule-based GTFS routing using actual departure/arrival times
2. ✅ Calendar support for date-specific service validation
3. ✅ Database-driven line-following algorithm (3.3x faster than graph approach)
4. ✅ Geocode German addresses using OpenStreetMap/Nominatim
5. ✅ Enhanced walking model: 20 minutes at start and end of journey
6. ✅ Line coverage optimization: Smart origin selection (10x speedup)
7. ✅ Interactive map visualization with OpenStreetMap tiles
8. ✅ Time-layered polygon overlays with realistic boundaries
9. ✅ **Enhanced UX**: Auto-selected date/time display and reference info in HTML maps
10. ✅ **Data Quality**: Import-time data cleaning and geographic bounds filtering
11. ✅ **Performance**: Strategic database indexes for 2-5x query speedup

**Schedule-Based Routing:**
- Uses actual GTFS stop_times data for precise departure/arrival times
- Calendar validation: Only includes trips that run on specified date
- Time-dependent Dijkstra: Finds earliest arrival times respecting schedules
- Real wait times: Calculates actual waiting at stops based on next departure
- Default: Monday April 7, 2025 at 8:00 AM (non-holiday weekday)

**Enhanced Walking Model:**
- Multi-origin routing from all stops within 20-minute walk
- Line coverage optimization: 320 → 30 origins while covering all 80 lines
- End walking expansion: Circle unions around transit destinations
- Time-budgeted walking: Only allows walking within remaining time

**Interactive Visualization Features:**
- Circle union boundaries: Precise walking areas computed as unions of circles around transit stops
- Time-layered overlays: Color-coded zones (0-10min, 10-20min, 20-30min) based on actual arrival times
- Interactive transit markers: Click/hover stops to see travel times and human-readable line names (447, U43, S1)
- Schedule information: Shows actual departure times and wait times
- **Reference date/time display**: HTML maps show reference date and departure time as subtitle
- Accurate geometry: Preserves holes, disconnected areas, and complex shapes
- Static HTML output: Self-contained files ready for GitHub Pages deployment

**Performance Optimizations:**
- **Strategic database indexes**: 11 indexes targeting hottest query paths for 2-5x speedup
- Calendar filtering: Only processes trips running on specified date
- Batch processing: Handles large GTFS datasets efficiently
- Line coverage optimization: Reduces routing origins by 10x
- Time window filtering: Only builds graph for relevant time periods
- Transfer optimization: Uses GTFS transfer data with realistic penalties
- **Import-time data quality**: Geographic bounds filtering and route name normalization during import
- Detailed progress reporting: Shows filtering, processing, and graph statistics

**Database-Driven Algorithm:**
- **Default algorithm**: The only routing algorithm in this branch
- **3.3x faster**: 17 seconds vs 56 seconds (compared to previous graph approach)
- **More comprehensive**: Finds 343 stops vs 190
- **Memory efficient**: No graph construction overhead
- **Simple logic**: Uses visited tracking and SQL queries
- **Line following**: Queries entire trip routes with single SQL call
- **Service filtering**: Uses calendar_dates table for active trips
- **Time window optimization**: Only considers trips within analysis window

**Latest Improvements (January 2025):**
- **Enhanced User Experience**:
  - Prominent auto-selected date/time display with green formatting
  - Reference date and departure time shown in generated HTML maps
  
- **Data Quality & Performance**:
  - Import-time data cleaning: Geographic bounds filtering, route name normalization
  - Strategic database indexes: 11 performance-critical indexes for 2-5x query speedup
  - Robust time format handling: Supports both HH:MM:SS and H:MM:SS formats
  - Multi-dataset support: VRR, DELFI, and HVV databases all fully supported

**Real-World Accuracy:**
- **Düsseldorf Hbf**: 771 reachable stops (schedule-based) vs 2,557 (estimates) 
- **Dortmund TU (graph)**: 190 reachable stops in 56 seconds
- **Dortmund TU (database)**: 343 reachable stops in 17 seconds
- Average wait times: 1-2 minutes (realistic) vs 0 (theoretical)
- Respects actual service patterns, holidays, and timetables