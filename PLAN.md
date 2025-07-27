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

### 1. Enhanced Walking Model (In Progress)
**Current Issue**: Walking is limited to 500m transfers between stops

**New Model**: Allow up to 20 minutes of walking time within total journey time

#### Walking Components:
1. **Start walking**: Walk up to 20min to reach any transit stop
2. **End walking**: Walk up to 20min from final transit stop
3. **Transfer walking**: Keep current 500m limit between stops

#### Implementation Plan:
- **Multi-origin Dijkstra**: Use ALL stops within 20min walk as starting points
- **Time budgeting**: For each origin stop, remaining_time = total_time - walk_time_to_stop  
- **End expansion**: Add 20min walking radius to all transit-reachable stops
- **Result display**: Show walking vs transit time breakdown

### 2. Spatial Optimization for Walking Connections
**Current Issue**: O(n²) walking connection calculation causes 2000-stop limit

**Solution**: Replace with spatial indexing
- Use R-tree or KD-tree for efficient nearest neighbor search
- Only calculate walking connections for stops actually within 500m
- Remove arbitrary 2000-stop performance limit

### 3. Visualization and Export
**Purpose**: Generate visual isochrone maps

- Add `folium` dependency for interactive web maps
- Generate isochrone polygons:
  - Convex hull or alpha shapes around reachable points (including walking expansion)
  - Multiple time rings (15min, 30min, 45min, etc.)
  - Color coding by travel time
- Export options:
  - Interactive HTML map with OpenStreetMap tiles
  - GeoJSON for use in other mapping tools
  - Static PNG images

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

2. **Phase 2** (In Progress): Enhanced walking model
   - Multi-origin Dijkstra algorithm
   - 20-minute walking time budget
   - End-of-journey walking expansion
   - Spatial indexing for transfers

3. **Phase 3**: Visualization and export
   - Interactive map generation
   - Isochrone polygon creation
   - GeoJSON/HTML export

4. **Phase 4**: Advanced features
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