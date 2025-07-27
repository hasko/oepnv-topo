# Development Plan: ÖPNV Travel Time Calculator

## Goal
Create a tool that calculates and visualizes how far you can travel using public transport (and walking) within a given time from any address in Germany.

## Current Status (Completed)

### ✅ Phase 1: Core Infrastructure 
- ✅ Geocoding with OpenStreetMap/Nominatim (cached, rate-limited)
- ✅ GTFS database import and optimization
- ✅ Basic isochrone calculation using Dijkstra's algorithm
- ✅ Optimized graph building with distance/journey filtering
- ✅ CLI interface with address and coordinate input

### ✅ Current Working Features
```bash
# Working commands
uv run python main.py init                                     # Import GTFS data
uv run python main.py stats                                    # Show database stats
uv run python main.py query --address "Düsseldorf Hbf" --time 30  # Isochrone calc
uv run python main.py query --lat 51.2197 --lon 6.7943 --time 20  # Coordinate input
```

## Next Steps

### 1. Enhanced Walking Model ✅ (Completed)
**New Model**: 20 minutes of walking time at start and end of journey

#### Walking Components:
1. ✅ **Start walking**: Walk up to 20min to reach any transit stop
2. ✅ **End walking**: Walk up to 20min from final transit stop  
3. ✅ **Transfer walking**: 500m limit between stops (realistic transfers)
4. ✅ **Line coverage optimization**: Smart origin selection to avoid redundancy

#### Implemented Features:
- ✅ **Multi-origin Dijkstra**: From optimized subset of walkable stops
- ✅ **Time budgeting**: remaining_time = total_time - walk_time_to_stop
- ✅ **End expansion**: 20min walking radius from all transit destinations
- ✅ **Line coverage**: 320 → 30 origins (10x speedup) while covering all 80 lines
- ✅ **Rich output**: Walking vs transit time breakdown with emoji indicators

#### Results:
- **Before**: ~5 reachable stops in 20 minutes
- **After**: 3,173 reachable points (125 transit + 3,048 walking destinations)

### 2. Spatial Optimization for Walking Connections
**Current Issue**: O(n²) walking connection calculation causes 2000-stop limit

**Solution**: Replace with spatial indexing
- Use R-tree or KD-tree for efficient nearest neighbor search
- Only calculate walking connections for stops actually within 500m
- Remove arbitrary 2000-stop performance limit

### 3. Visualization and Export ✅ (Completed)
**Purpose**: Generate visual isochrone maps

#### Implemented Features:
- ✅ **Interactive web maps**: Folium-based with OpenStreetMap tiles
- ✅ **Alpha shape polygons**: Realistic boundaries around reachable points
- ✅ **Time-layered visualization**: Color-coded zones (0-10min, 10-20min, 20-30min, etc.)
- ✅ **CLI integration**: Both standalone and integrated with query command
- ✅ **Transit stop markers**: Individual stops with travel time information
- ✅ **Professional styling**: Legend, tooltips, and responsive design

#### Available Commands:
```bash
# Standalone visualization
uv run python main.py visualize --address "Düsseldorf Hbf" --time 30

# Simple boundary mode
uv run python main.py visualize --lat 51.2197 --lon 6.7943 --time 20 --simple

# Integrated with query
uv run python main.py query --address "Düsseldorf Hbf" --time 20 --visualize
```

#### Pending Export Options:
- GeoJSON export for use in other mapping tools
- Static PNG image generation

### 4. Time-Dependent Routing (Future)
**Purpose**: Use actual schedules instead of simplified wait times

- Parse GTFS calendar data for service availability
- Consider actual departure times based on time of day
- Handle frequency-based vs schedule-based routes
- Account for weekend/holiday schedule variations

### 5. Advanced Features (Future)
- **Multi-modal routing**: Include bike sharing, Park+Ride
- **Accessibility**: Wheelchair-accessible routes only
- **Real-time integration**: Delays and service disruptions
- **Batch processing**: Generate isochrones for multiple origins
- **API endpoint**: REST API for web application integration

## Implementation Order

1. **Phase 1** (Completed): Basic infrastructure ✓
   - CLI framework ✓
   - Database import ✓
   - Geocoding ✓
   - Graph construction ✓
   - Basic isochrone calculation ✓

2. **Phase 2** (Completed): Enhanced walking model ✓
   - Multi-origin Dijkstra algorithm ✓
   - 20-minute walking time budget ✓
   - End-of-journey walking expansion ✓
   - Line coverage optimization ✓

3. **Phase 3** (Completed): Visualization and export ✓
   - Interactive map generation ✓
   - Isochrone polygon creation ✓
   - HTML export ✓
   - CLI integration ✓
   - Color optimization ✓

4. **Phase 4** (Completed): Bug fixes and graph optimization ✓
   - ✅ Walking time calculation fix
   - ✅ Debug tools for route investigation  
   - ✅ Improved visualization colors (magenta/purple)
   - ✅ Fixed missing bus connections (Route 447, 440, X13)
   - ✅ Eliminated incorrect wait times for same-route connections
   - ✅ Added proper transfer penalties using GTFS transfer data
   - ✅ Implemented direction filtering to prevent U-turns (~46% reduction)
   - ✅ Enhanced progress reporting with detailed connection statistics

5. **Phase 5**: Advanced features
   - Time-dependent routing
   - Multi-modal integration
   - Real-time data

## Technical Decisions

### Why SQLite?
- No server required
- Sufficient for ~6M records
- Good spatial query support
- Easy distribution

### Why NetworkX?
- Mature graph library
- Built-in shortest path algorithms
- Easy to extend for custom routing

### Why OpenStreetMap/Nominatim?
- Free and open data
- Excellent coverage in Germany
- No API key required
- Respects user privacy
- Active community maintaining data quality

### Time Complexity
- Graph build: O(n) where n = number of connections
- Single query: O((V + E) log V) with Dijkstra's
- Can be improved to O(V log V) with preprocessing

## Future Enhancements

1. **Multi-modal routing**: Include bike sharing, car parks
2. **Accessibility**: Wheelchair-accessible routes only
3. **Real-time data**: Integrate delays and disruptions
4. **API endpoint**: REST API for web integration
5. **Batch processing**: Generate isochrones for entire cities
6. **Time windows**: "Arrive by" vs "Depart at"