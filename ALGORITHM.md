# Database-Driven Line-Following Algorithm

## Overview

The database-driven line-following algorithm (`database_line_following_isochrone`) is an efficient approach to finding all transit stops reachable within a given time budget from a starting location. It uses direct SQL queries to explore the transit network without building a graph structure in memory.

## Core Concept

Instead of constructing a time-expanded graph with nodes for each (stop, time) pair, this algorithm:
- Uses the database as the primary data structure
- Performs breadth-first search with SQL queries
- Follows entire transit lines in single database operations
- Tracks visited stops to avoid redundant exploration

## Algorithm Flow

### 1. Initialization
- **Input**: Starting location, time budget, departure time, date
- **Walkable Stops**: Find all stops within 20-minute walk from origin
- **Time Window**: Calculate valid time range (start_time to start_time + max_time)
- **Service Validation**: Applied directly in SQL queries using calendar_dates table (no pre-loading)
- **Queue Setup**: Initialize exploration queue with walkable stops and their arrival times

### 2. Main Exploration Loop

The algorithm uses a simple queue-based approach:

```
while queue is not empty:
    1. Dequeue next stop (current_stop, arrival_time, total_time)
    
    2. Skip if already visited with better time
    
    3. Mark as visited with current arrival time
    
    4. Find all trips departing after arrival_time:
       - Query departing trips from current_stop
       - Filter by time window and active services
    
    5. For each departing trip:
       - Get ALL downstream stops on the trip
       - Add unvisited stops to queue
    
    6. Check for transfers:
       - Query direct transfers from current_stop
       - Add transfer destinations to queue
```

### 3. Key Features

- **Visited Tracking**: Maintains dictionary of stop_id → earliest_arrival_time
- **Line Following**: Gets entire trip route with one query (not stop-by-stop)
- **Transfer Support**: Includes walking transfers between nearby stops
- **Time Validation**: Ensures all arrivals fit within time budget

## Key SQL Queries

### 1. Query Departing Trips (`query_departing_trips_simple`)

```sql
SELECT DISTINCT
    st.trip_id,
    st.departure_time,
    st.stop_sequence,
    r.route_short_name
FROM stop_times st
JOIN trips t ON st.trip_id = t.trip_id
JOIN routes r ON t.route_id = r.route_id
WHERE st.stop_id = ?
  AND st.departure_time >= ?
  AND st.departure_time <= ?
  AND t.service_id IN (
      SELECT service_id FROM calendar_dates 
      WHERE date = ? AND exception_type = 1
  )
  AND EXISTS (
      -- Ensure trip has at least one stop within our time window
      SELECT 1 FROM stop_times st2
      WHERE st2.trip_id = st.trip_id
      AND st2.arrival_time <= ?
  )
ORDER BY st.departure_time
LIMIT 100
```

**Purpose**: Find trips leaving a stop within the time window on the specific date, filtering out trips that end before our time window

### 2. Query All Stops on Trip (`query_all_stops_on_trip`)

```sql
SELECT stop_id, arrival_time
FROM stop_times
WHERE trip_id = ?
  AND stop_sequence > ?
  AND arrival_time <= ?
ORDER BY stop_sequence
```

**Purpose**: Get all reachable stops downstream on a specific trip

### 3. Query Direct Transfers (`query_direct_transfers`)

```sql
SELECT to_stop_id, min_transfer_time
FROM transfers
WHERE from_stop_id = ?
  AND from_stop_id != to_stop_id
LIMIT 20
```

**Purpose**: Find walking transfer opportunities from current stop

## Performance Advantages

### 1. No Graph Construction Overhead
- Avoids building time-expanded graph with thousands of nodes
- No memory allocation for graph edges
- Direct database queries instead of graph traversal

### 2. Efficient Line Following
- Single query retrieves entire trip route
- Batch processing of downstream stops
- Reduces number of database round-trips

### 3. Smart Pruning
- Visited tracking prevents redundant exploration
- Time window filtering in SQL reduces data transfer
- Service filtering uses indexed calendar tables
- Trip filtering excludes trips that end before time window (optimization)

### 4. Results
- **3.3x faster**: 17 seconds vs 56 seconds (graph approach)
- **More comprehensive**: Finds 343 stops vs 190 (Dortmund TU example)
- **Memory efficient**: Minimal in-memory data structures

## Comparison with Graph-Based Approach

### Database-Driven Advantages
- Simpler implementation (BFS vs Dijkstra)
- Better performance on large datasets
- Lower memory footprint
- More complete results (finds more reachable stops)

### Graph-Based Advantages
- More flexible for complex routing rules
- Better for multi-criteria optimization
- Easier to implement advanced features (e.g., fare zones)
- Can handle dynamic costs more easily

### When to Use Each
- **Database-Driven**: Best for isochrone/reachability analysis
- **Graph-Based**: Better for point-to-point routing with complex constraints

## Implementation Notes

### Service Filtering
The algorithm uses GTFS calendar_dates table for date-specific filtering:
- `exception_type = 1`: Service is added on this date
- `exception_type = 2`: Service is removed on this date
- No pre-loading of active trips - filtering happens directly in SQL queries
- Time window filtering ensures only relevant trips are considered

### Time Handling
- All times stored as strings in GTFS format (HH:MM:SS)
- Converted to seconds for calculations
- Handles times past midnight (e.g., "25:30:00")

### Transfer Handling
- Default transfer time: 5 minutes
- Only explores transfers if sufficient time remains
- Uses GTFS transfers table when available

## Optimization Opportunities

1. **Parallel Processing**: Multiple starting stops could be explored in parallel
2. **Caching**: Frequently used trips/stops could be cached
3. **Index Optimization**: Database indexes on (stop_id, departure_time)
4. **Batch Queries**: Combine multiple stop queries when possible