#!/usr/bin/env python3
"""
ÖPNV Topo - Topological coloring based on public transport travel durations
"""
import click
import sqlite3
import csv
import os
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
import pandas as pd
from math import radians, cos, sin, acos, sqrt, atan2
from typing import Dict, Tuple, List, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import json
from datetime import datetime
import networkx as nx
from visualization import create_isochrone_map, create_simple_boundary_map


console = Console()

# Cache file for geocoding results
GEOCODE_CACHE_FILE = ".geocode_cache.json"


def load_geocode_cache() -> Dict:
    """Load geocoding cache from file"""
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            with open(GEOCODE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_geocode_cache(cache: Dict):
    """Save geocoding cache to file"""
    with open(GEOCODE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def geocode_address(address: str, country_code: str = 'de') -> Optional[Tuple[float, float]]:
    """
    Geocode an address using OpenStreetMap's Nominatim service
    Returns (latitude, longitude) or None if not found
    """
    # Check cache first
    cache = load_geocode_cache()
    cache_key = f"{address}_{country_code}"
    
    if cache_key in cache:
        console.print(f"[dim]Using cached coordinates for {address}[/dim]")
        return tuple(cache[cache_key])
    
    # Initialize Nominatim with proper user agent
    geolocator = Nominatim(user_agent="oepnv-topo/0.1.0 (https://github.com/user/oepnv-topo)")
    
    try:
        # Rate limiting - Nominatim requires max 1 request per second
        time.sleep(1)
        
        # Search with country restriction
        location = geolocator.geocode(
            address, 
            country_codes=[country_code],
            language='de',
            timeout=10
        )
        
        if location:
            coords = (location.latitude, location.longitude)
            # Cache the result
            cache[cache_key] = coords
            save_geocode_cache(cache)
            
            console.print(f"[green]Found: {location.address}[/green]")
            console.print(f"[dim]Coordinates: {coords[0]:.6f}, {coords[1]:.6f}[/dim]")
            console.print("[dim]Data © OpenStreetMap contributors[/dim]")
            return coords
        else:
            console.print(f"[yellow]Address not found: {address}[/yellow]")
            return None
            
    except GeocoderTimedOut:
        console.print("[red]Geocoding request timed out. Please try again.[/red]")
        return None
    except GeocoderServiceError as e:
        console.print(f"[red]Geocoding service error: {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Unexpected error during geocoding: {e}[/red]")
        return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return r * c


def find_nearest_stops(conn: sqlite3.Connection, lat: float, lon: float, 
                      max_distance_km: float = 5.0, limit: int = 10) -> List[Dict]:
    """Find the nearest stops to a given location"""
    cursor = conn.cursor()
    
    # Approximate degrees for bounding box
    lat_delta = max_distance_km / 111.0
    lon_delta = max_distance_km / (111.0 * cos(radians(lat)))
    
    query = """
    WITH bounded_stops AS (
        SELECT stop_id, stop_name, stop_lat, stop_lon
        FROM stops
        WHERE stop_lat BETWEEN ? AND ?
          AND stop_lon BETWEEN ? AND ?
    )
    SELECT stop_id, stop_name, stop_lat, stop_lon
    FROM bounded_stops
    ORDER BY (stop_lat - ?) * (stop_lat - ?) + (stop_lon - ?) * (stop_lon - ?)
    LIMIT ?
    """
    
    cursor.execute(query, (
        lat - lat_delta, lat + lat_delta,
        lon - lon_delta, lon + lon_delta,
        lat, lat, lon, lon,
        limit * 2  # Get more candidates for accurate distance calculation
    ))
    
    results = []
    for row in cursor.fetchall():
        stop_id, stop_name, stop_lat, stop_lon = row
        distance = haversine_distance(lat, lon, stop_lat, stop_lon)
        
        if distance <= max_distance_km:
            results.append({
                'stop_id': stop_id,
                'stop_name': stop_name,
                'stop_lat': stop_lat,
                'stop_lon': stop_lon,
                'distance_km': distance
            })
    
    # Sort by actual distance and limit
    results.sort(key=lambda x: x['distance_km'])
    return results[:limit]


def find_walkable_stops(conn: sqlite3.Connection, start_lat: float, start_lon: float, 
                       max_walk_time_minutes: int = 20) -> List[Dict]:
    """
    Find all stops within walking distance of starting point
    Returns list with stop info and walking time to reach each stop
    """
    max_walk_distance_km = (max_walk_time_minutes / 60.0) * 5.0  # 5 km/h walking speed
    
    cursor = conn.cursor()
    cursor.execute("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops")
    
    walkable_stops = []
    for stop_id, stop_name, stop_lat, stop_lon in cursor.fetchall():
        distance_km = haversine_distance(start_lat, start_lon, stop_lat, stop_lon)
        
        if distance_km <= max_walk_distance_km:
            walk_time_minutes = (distance_km / 5.0) * 60  # Convert back to minutes
            walkable_stops.append({
                'stop_id': stop_id,
                'stop_name': stop_name,
                'stop_lat': stop_lat,
                'stop_lon': stop_lon,
                'distance_km': distance_km,
                'walk_time_minutes': walk_time_minutes
            })
    
    # Sort by walking time
    walkable_stops.sort(key=lambda x: x['walk_time_minutes'])
    return walkable_stops


def get_stop_lines_mapping(conn: sqlite3.Connection) -> Dict[str, set]:
    """
    Create mapping of stop_id -> set of route_ids that serve this stop
    This is used for line coverage optimization
    """
    cursor = conn.cursor()
    
    # Query to find which routes serve each stop
    query = """
    SELECT DISTINCT st.stop_id, t.route_id
    FROM stop_times st
    JOIN trips t ON st.trip_id = t.trip_id
    WHERE st.stop_id IS NOT NULL AND t.route_id IS NOT NULL
    """
    
    stop_lines = {}
    cursor.execute(query)
    
    for stop_id, route_id in cursor.fetchall():
        if stop_id not in stop_lines:
            stop_lines[stop_id] = set()
        stop_lines[stop_id].add(route_id)
    
    return stop_lines


def optimize_walkable_stops_by_line_coverage(walkable_stops: List[Dict], 
                                           stop_lines_mapping: Dict[str, set]) -> List[Dict]:
    """
    Use greedy algorithm to select minimal set of walkable stops that covers all reachable lines
    
    Only include a stop if it provides access to lines not available from closer stops
    """
    console.print("[cyan]Optimizing walkable stops using line coverage...[/cyan]")
    
    covered_lines = set()
    selected_stops = []
    
    # Sort by walking time (closest first)
    sorted_stops = sorted(walkable_stops, key=lambda x: x['walk_time_minutes'])
    
    for stop in sorted_stops:
        stop_id = stop['stop_id']
        
        # Get lines served by this stop
        stop_lines = stop_lines_mapping.get(stop_id, set())
        
        if not stop_lines:
            # Skip stops with no line information
            continue
        
        # Check if this stop offers any new lines
        new_lines = stop_lines - covered_lines
        
        if new_lines:
            # This stop provides access to lines we haven't covered yet
            selected_stops.append(stop)
            covered_lines.update(new_lines)
            
            # Add debug info about what lines this stop provides
            stop['new_lines'] = new_lines
            stop['all_lines'] = stop_lines
    
    console.print(f"[green]Line coverage optimization: {len(walkable_stops)} → {len(selected_stops)} stops[/green]")
    console.print(f"[dim]Covers {len(covered_lines)} unique lines/routes[/dim]")
    
    # Show first few selected stops with their line contributions
    console.print("[dim]Selected stops and their unique line contributions:[/dim]")
    for i, stop in enumerate(selected_stops[:5]):
        lines_str = ', '.join(sorted(list(stop['new_lines']))[:3])
        if len(stop['new_lines']) > 3:
            lines_str += f" (+{len(stop['new_lines'])-3} more)"
        console.print(f"  {stop['stop_name'][:30]} ({stop['walk_time_minutes']:.1f}min): {lines_str}")
    
    if len(selected_stops) > 5:
        console.print(f"  ... and {len(selected_stops) - 5} more stops")
    
    return selected_stops


def calculate_multi_origin_isochrone(graph: nx.DiGraph, walkable_stops: List[Dict], 
                                   max_total_time_minutes: int) -> Dict[str, Dict]:
    """
    Calculate isochrone using multiple starting points (all walkable stops)
    Returns dict mapping stop_id to best travel info (time, route, etc.)
    """
    max_total_time_seconds = max_total_time_minutes * 60
    best_times = {}  # stop_id -> {total_time, walk_time, transit_time, origin_stop}
    
    console.print(f"[cyan]Running multi-origin Dijkstra from {len(walkable_stops)} walkable stops...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing origins...", total=len(walkable_stops))
        
        for walkable_stop in walkable_stops:
            origin_stop_id = walkable_stop['stop_id']
            walk_time_seconds = walkable_stop['walk_time_minutes'] * 60
            available_transit_time = max_total_time_seconds - walk_time_seconds
            
            # Skip if walking time alone exceeds total time limit
            if available_transit_time <= 0:
                progress.update(task, advance=1)
                continue
            
            # Skip if this stop is not in the graph
            if not graph.has_node(origin_stop_id):
                progress.update(task, advance=1)
                continue
            
            # Run Dijkstra from this origin with remaining time budget
            try:
                reachable_from_origin = nx.single_source_dijkstra_path_length(
                    graph, origin_stop_id, cutoff=available_transit_time, weight='weight'
                )
                
                # Update best times for each reachable stop
                for stop_id, transit_time_seconds in reachable_from_origin.items():
                    total_time_seconds = walk_time_seconds + transit_time_seconds
                    
                    if total_time_seconds <= max_total_time_seconds:
                        # Keep this result if it's better than previous best
                        if (stop_id not in best_times or 
                            total_time_seconds < best_times[stop_id]['total_time_seconds']):
                            
                            best_times[stop_id] = {
                                'total_time_seconds': total_time_seconds,
                                'walk_time_seconds': walk_time_seconds,
                                'transit_time_seconds': transit_time_seconds,
                                'origin_stop': walkable_stop,
                                'total_time_minutes': total_time_seconds / 60.0,
                                'walk_time_minutes': walk_time_seconds / 60.0,
                                'transit_time_minutes': transit_time_seconds / 60.0
                            }
            except:
                # Skip problematic origins
                pass
            
            progress.update(task, advance=1)
    
    return best_times


def add_end_walking_expansion(reachable_stops: Dict[str, Dict], graph: nx.DiGraph, 
                            max_walk_time_minutes: int = 20, max_total_time_minutes: int = 60) -> Dict[str, Dict]:
    """
    For each transit-reachable stop, add all points within walking distance
    This represents walking from the final transit stop to the destination
    
    Returns expanded reachable points with coordinates
    """
    max_walk_distance_km = (max_walk_time_minutes / 60.0) * 5.0  # 5 km/h walking speed
    expanded_points = {}
    
    console.print(f"[cyan]Adding end-of-journey walking expansion (up to {max_walk_time_minutes} min)...[/cyan]")
    
    for stop_id, stop_info in reachable_stops.items():
        # Get stop coordinates from graph
        if not graph.has_node(stop_id):
            continue
            
        stop_data = graph.nodes[stop_id]
        stop_lat = stop_data['lat']
        stop_lon = stop_data['lon']
        
        # The reachable point is the stop itself
        point_key = f"stop_{stop_id}"
        if point_key not in expanded_points or stop_info['total_time_minutes'] < expanded_points[point_key]['total_time_minutes']:
            expanded_points[point_key] = {
                **stop_info,
                'point_type': 'transit_stop',
                'lat': stop_lat,
                'lon': stop_lon,
                'name': stop_data['name'],
                'end_walk_time_minutes': 0.0
            }
        
        # Add walking radius around this stop
        # For visualization, we'll add sample points in a grid pattern
        walk_radius_deg = max_walk_distance_km / 111.0  # Rough conversion to degrees
        num_points_per_side = 5  # Create a 5x5 grid of sample points
        
        for i in range(num_points_per_side):
            for j in range(num_points_per_side):
                # Create grid point within walking radius
                lat_offset = (i - num_points_per_side//2) * (2 * walk_radius_deg) / num_points_per_side
                lon_offset = (j - num_points_per_side//2) * (2 * walk_radius_deg) / num_points_per_side
                
                point_lat = stop_lat + lat_offset
                point_lon = stop_lon + lon_offset
                
                # Calculate actual walking distance
                walk_distance_km = haversine_distance(stop_lat, stop_lon, point_lat, point_lon)
                
                if walk_distance_km <= max_walk_distance_km:
                    walk_time_minutes = (walk_distance_km / 5.0) * 60
                    
                    # BUG FIX: Don't add walking time to total - the time budget should already include it
                    # The original transit time to reach this stop already accounts for the time budget
                    # We only add walking destinations that can be reached within the original time limit
                    
                    # Calculate what the total time would be if we walked to this point
                    total_time_would_be = stop_info['total_time_minutes'] + walk_time_minutes
                    
                    # BUG FIX: Only include walking destinations that fit within the original time budget
                    # This prevents final walking time from being added on top of the time limit
                    if total_time_would_be <= max_total_time_minutes:
                        point_key = f"walk_{point_lat:.4f}_{point_lon:.4f}"
                        
                        # Keep this point if it's better than existing or new
                        if (point_key not in expanded_points or 
                            total_time_would_be < expanded_points[point_key]['total_time_minutes']):
                            
                            expanded_points[point_key] = {
                                'total_time_minutes': total_time_would_be,  # Store actual total time
                                'point_type': 'walking_destination',
                                'lat': point_lat,
                                'lon': point_lon,
                                'name': f"Walking from {stop_data['name'][:20]}",
                                'transit_stop_info': stop_info,
                                'end_walk_time_minutes': walk_time_minutes,
                                'walk_distance_km': walk_distance_km,
                                'transit_time_to_stop': stop_info['total_time_minutes']  # Track transit time separately
                            }
    
    console.print(f"[green]Expanded to {len(expanded_points)} reachable points (including walking destinations)[/green]")
    return expanded_points


def build_transit_graph(conn: sqlite3.Connection, 
                       start_lat: float, start_lon: float, 
                       max_time_minutes: int = 30,
                       max_walking_distance_m: int = 500) -> nx.DiGraph:
    """
    Build a directed graph representing the transit network
    
    Nodes: stops with attributes (lat, lon, name)
    Edges: connections with weights representing travel time in seconds
    
    Optimized for a specific starting location and maximum travel time
    """
    # Calculate maximum reachable distance (assuming 50 km/h max speed)
    max_distance_km = (max_time_minutes / 60.0) * 50
    max_stops_per_trip = max_time_minutes  # 1 stop per minute maximum
    
    console.print(f"[cyan]Building optimized transit network graph...[/cyan]")
    console.print(f"[dim]Start: {start_lat:.4f}, {start_lon:.4f}[/dim]")
    console.print(f"[dim]Max time: {max_time_minutes}min, Max distance: {max_distance_km:.1f}km[/dim]")
    console.print(f"[dim]Max stops per trip: {max_stops_per_trip}[/dim]")
    
    G = nx.DiGraph()
    cursor = conn.cursor()
    
    # Add only reachable stops as nodes
    console.print("[cyan]Adding reachable stops as nodes...[/cyan]")
    cursor.execute("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops")
    
    reachable_stops = []
    for stop_id, stop_name, stop_lat, stop_lon in cursor.fetchall():
        distance_km = haversine_distance(start_lat, start_lon, stop_lat, stop_lon)
        
        if distance_km <= max_distance_km:
            G.add_node(stop_id, 
                      name=stop_name, 
                      lat=stop_lat, 
                      lon=stop_lon,
                      distance_from_start=distance_km)
            reachable_stops.append(stop_id)
    
    console.print(f"[green]Added {G.number_of_nodes():,} reachable stops (filtered from 34k+)[/green]")
    
    # Add transit connections from stop_times
    console.print("[cyan]Adding transit connections...[/cyan]")
    
    # Skip if no reachable stops found
    if not reachable_stops:
        console.print("[yellow]No reachable stops found within distance limit[/yellow]")
        return G
    
    # Create a temporary table with reachable stops for faster filtering
    reachable_stops_str = "'" + "','".join(reachable_stops) + "'"
    
    # Query to get consecutive stops on each trip with times, filtered by reachable stops
    query = f"""
    WITH consecutive_stops AS (
        SELECT 
            st1.trip_id,
            st1.stop_id as from_stop,
            st2.stop_id as to_stop,
            st1.departure_time,
            st2.arrival_time,
            st1.stop_sequence,
            st2.stop_sequence as next_sequence
        FROM stop_times st1
        JOIN stop_times st2 ON st1.trip_id = st2.trip_id 
                           AND st2.stop_sequence = st1.stop_sequence + 1
        WHERE st1.departure_time IS NOT NULL 
          AND st2.arrival_time IS NOT NULL
          AND st1.departure_time != ''
          AND st2.arrival_time != ''
          AND st1.stop_id IN ({reachable_stops_str})
          AND st2.stop_id IN ({reachable_stops_str})
          AND st1.stop_sequence <= {max_stops_per_trip}
          AND st2.stop_sequence <= {max_stops_per_trip}
    )
    SELECT from_stop, to_stop, departure_time, arrival_time, COUNT(*) as frequency
    FROM consecutive_stops
    GROUP BY from_stop, to_stop, departure_time, arrival_time
    """
    
    # Process in chunks to handle large dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing transit connections...")
        
        cursor.execute(query)
        connection_count = 0
        
        for row in cursor.fetchall():
            from_stop, to_stop, dep_time, arr_time, frequency = row
            
            try:
                # Parse times (format: HH:MM:SS)
                dep_hours, dep_mins, dep_secs = map(int, dep_time.split(':'))
                arr_hours, arr_mins, arr_secs = map(int, arr_time.split(':'))
                
                dep_seconds = dep_hours * 3600 + dep_mins * 60 + dep_secs
                arr_seconds = arr_hours * 3600 + arr_mins * 60 + arr_secs
                
                # Handle day overflow (times > 24:00:00)
                if arr_seconds < dep_seconds:
                    arr_seconds += 24 * 3600
                
                travel_time = arr_seconds - dep_seconds
                
                # Only add positive travel times
                if travel_time > 0:
                    # Use frequency as a measure of service quality (higher frequency = lower effective wait time)
                    # Add a base wait time of 5 minutes, reduced by frequency
                    wait_time = max(300 - frequency * 30, 60)  # Min 1 minute wait
                    total_time = travel_time + wait_time
                    
                    # Add edge if both stops exist in graph
                    if G.has_node(from_stop) and G.has_node(to_stop):
                        G.add_edge(from_stop, to_stop, 
                                 weight=total_time,
                                 travel_time=travel_time,
                                 wait_time=wait_time,
                                 frequency=frequency,
                                 type='transit')
                        connection_count += 1
                        
            except (ValueError, AttributeError):
                # Skip malformed time entries
                continue
                
            if connection_count % 10000 == 0:
                progress.update(task, description=f"Processed {connection_count:,} connections...")
    
    console.print(f"[green]Added {connection_count:,} transit connections[/green]")
    
    # Add walking connections between nearby stops (only among reachable stops)
    console.print(f"[cyan]Adding walking connections (max {max_walking_distance_m}m)...[/cyan]")
    
    walking_connections = 0
    stops_list = list(G.nodes(data=True))
    
    # Only process walking connections if we have a reasonable number of stops
    if len(stops_list) > 2000:
        console.print(f"[yellow]Skipping walking connections due to large number of stops ({len(stops_list):,})[/yellow]")
        console.print("[dim]Consider reducing max distance or time to enable walking connections[/dim]")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Adding walking connections...", total=len(stops_list))
            
            for i, (stop1_id, stop1_data) in enumerate(stops_list):
                for stop2_id, stop2_data in stops_list[i+1:]:
                    distance_m = haversine_distance(
                        stop1_data['lat'], stop1_data['lon'],
                        stop2_data['lat'], stop2_data['lon']
                    ) * 1000
                    
                    if distance_m <= max_walking_distance_m:
                        # Walking speed: ~5 km/h = ~1.4 m/s
                        walk_time = distance_m / 1.4
                        
                        # Add bidirectional walking edges
                        G.add_edge(stop1_id, stop2_id, 
                                 weight=walk_time,
                                 distance_m=distance_m,
                                 type='walking')
                        G.add_edge(stop2_id, stop1_id, 
                                 weight=walk_time,
                                 distance_m=distance_m,
                                 type='walking')
                        walking_connections += 2
                
                progress.update(task, advance=1)
    
    console.print(f"[green]Added {walking_connections:,} walking connections[/green]")
    console.print(f"[bold green]Graph completed: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges[/bold green]")
    
    return G


def save_graph(graph: nx.DiGraph, filename: str = "transit_graph.gpickle"):
    """Save the graph to disk for faster loading"""
    console.print(f"[cyan]Saving graph to {filename}...[/cyan]")
    nx.write_gpickle(graph, filename)
    console.print(f"[green]Graph saved successfully[/green]")


def load_graph(filename: str = "transit_graph.gpickle") -> Optional[nx.DiGraph]:
    """Load a previously saved graph"""
    if os.path.exists(filename):
        console.print(f"[cyan]Loading graph from {filename}...[/cyan]")
        graph = nx.read_gpickle(filename)
        console.print(f"[green]Loaded graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges[/green]")
        return graph
    return None


@click.group()
def cli():
    """ÖPNV Topo - Public transport topology mapping tool"""
    pass


@cli.command()
@click.option('--db-path', default='oepnv.db', help='Path to SQLite database file')
@click.option('--data-dir', default='data', help='Directory containing GTFS data files')
def init(db_path: str, data_dir: str):
    """Initialize database with GTFS data from the data folder"""
    console.print(f"[bold blue]Initializing ÖPNV database at {db_path}[/bold blue]")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        console.print(f"[yellow]Removing existing database {db_path}[/yellow]")
        os.remove(db_path)
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create database schema
        create_schema(cursor)
        
        # Import data files
        stats = import_gtfs_data(conn, data_dir)
        
        # Display summary statistics
        display_summary(stats)
        
        conn.commit()
        console.print(f"[bold green]Database initialized successfully at {db_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during initialization: {e}[/bold red]")
        conn.rollback()
        raise
    finally:
        conn.close()


def create_schema(cursor: sqlite3.Cursor):
    """Create database schema for GTFS data"""
    console.print("[cyan]Creating database schema...[/cyan]")
    
    # Stops table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stops (
            stop_id TEXT PRIMARY KEY,
            stop_code TEXT,
            stop_name TEXT NOT NULL,
            stop_lat REAL NOT NULL,
            stop_lon REAL NOT NULL,
            stop_url TEXT,
            location_type INTEGER DEFAULT 0,
            parent_station TEXT,
            wheelchair_boarding INTEGER,
            platform_code TEXT,
            NVBW_HST_DHID TEXT
        )
    """)
    
    # Routes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS routes (
            route_id TEXT PRIMARY KEY,
            agency_id TEXT,
            route_short_name TEXT,
            route_long_name TEXT,
            route_type INTEGER NOT NULL,
            route_color TEXT,
            route_text_color TEXT,
            NVBW_DLID TEXT
        )
    """)
    
    # Trips table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            trip_id TEXT PRIMARY KEY,
            route_id TEXT NOT NULL,
            service_id TEXT NOT NULL,
            trip_headsign TEXT,
            trip_short_name TEXT,
            direction_id INTEGER,
            shape_id TEXT,
            wheelchair_accessible INTEGER,
            FOREIGN KEY (route_id) REFERENCES routes(route_id)
        )
    """)
    
    # Stop times table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stop_times (
            trip_id TEXT NOT NULL,
            arrival_time TEXT,
            departure_time TEXT,
            stop_id TEXT NOT NULL,
            stop_sequence INTEGER NOT NULL,
            stop_headsign TEXT,
            pickup_type INTEGER DEFAULT 0,
            drop_off_type INTEGER DEFAULT 0,
            shape_dist_traveled REAL,
            PRIMARY KEY (trip_id, stop_sequence),
            FOREIGN KEY (trip_id) REFERENCES trips(trip_id),
            FOREIGN KEY (stop_id) REFERENCES stops(stop_id)
        )
    """)
    
    # Transfers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            from_stop_id TEXT NOT NULL,
            to_stop_id TEXT NOT NULL,
            transfer_type INTEGER NOT NULL,
            min_transfer_time INTEGER,
            PRIMARY KEY (from_stop_id, to_stop_id),
            FOREIGN KEY (from_stop_id) REFERENCES stops(stop_id),
            FOREIGN KEY (to_stop_id) REFERENCES stops(stop_id)
        )
    """)
    
    # German-specific tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS haltestellen (
            VERSION TEXT,
            STOP_NR TEXT PRIMARY KEY,
            STOP_TYPE TEXT,
            STOP_NAME TEXT,
            STOP_NAME_WITHOUT_LOCALITY TEXT,
            STOP_SHORTNAME TEXT,
            STOP_POS_X TEXT,
            STOP_POS_Y TEXT,
            PLACE TEXT,
            OCC TEXT,
            FARE_ZONE1_NR TEXT,
            FARE_ZONE2_NR TEXT,
            FARE_ZONE3_NR TEXT,
            FARE_ZONE4_NR TEXT,
            FARE_ZONE5_NR TEXT,
            FARE_ZONE6_NR TEXT,
            GLOBAL_ID TEXT,
            VALID_FROM TEXT,
            VALID_TO TEXT,
            PLACE_ID TEXT,
            GIS_MOT_FLAG TEXT,
            IS_CENTRAL_STOP TEXT,
            IS_RESPONSIBLE_STOP TEXT,
            INTERCHANGE_TYPE TEXT,
            INTERCHANGE_QUALITY TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS linien (
            VERSION TEXT,
            BRANCH_NR TEXT,
            LINE_NR TEXT,
            STR_LINE_VAR TEXT,
            LINE_NAME TEXT,
            LINE_DIR_NR TEXT,
            LAST_MODIFIED TEXT,
            MOT_NR TEXT,
            VALID_FROM TEXT,
            VALID_TO TEXT,
            OP_CODE TEXT,
            OBO_SHORT_NAME TEXT,
            ROUTE_TYPE TEXT,
            GLOBAL_ID TEXT,
            BIKE_RULE TEXT,
            LINE_SPECIAL_FARE TEXT
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_lat_lon ON stops(stop_lat, stop_lon)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_stop_id ON stop_times(stop_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trips_route_id ON trips(route_id)")
    
    console.print("[green]Schema created successfully[/green]")


def import_gtfs_data(conn: sqlite3.Connection, data_dir: str) -> Dict[str, int]:
    """Import GTFS data from CSV files"""
    stats = {}
    data_path = Path(data_dir)
    
    # Define file mappings with their table names and encoding
    file_mappings = [
        ('stops.csv', 'stops', ',', 'utf-8'),
        ('routes.csv', 'routes', ',', 'utf-8'),
        ('trips.csv', 'trips', ',', 'utf-8'),
        ('stop_times.csv', 'stop_times', ',', 'utf-8'),
        ('transfers.csv', 'transfers', ',', 'utf-8'),
        ('haltestellen.csv', 'haltestellen', ';', 'latin-1'),
        ('linien.csv', 'linien', ';', 'latin-1'),
    ]
    
    for filename, table_name, delimiter, encoding in file_mappings:
        file_path = data_path / filename
        if file_path.exists():
            console.print(f"[cyan]Importing {filename}...[/cyan]")
            count = import_csv_file(conn, file_path, table_name, delimiter, encoding)
            stats[table_name] = count
            console.print(f"[green]Imported {count:,} records from {filename}[/green]")
        else:
            console.print(f"[yellow]Warning: {filename} not found, skipping[/yellow]")
    
    return stats


def import_csv_file(conn: sqlite3.Connection, file_path: Path, table_name: str, 
                   delimiter: str, encoding: str) -> int:
    """Import a single CSV file into the database"""
    cursor = conn.cursor()
    
    # Get file size for progress bar
    file_size = file_path.stat().st_size
    
    # Read CSV header to get column names
    with open(file_path, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = reader.fieldnames
        
        # Create placeholder string for SQL
        placeholders = ','.join(['?' for _ in columns])
        column_names = ','.join(columns)
        
        insert_sql = f"INSERT OR REPLACE INTO {table_name} ({column_names}) VALUES ({placeholders})"
        
        # Count total rows for large files
        if table_name == 'stop_times':
            # For stop_times, use pandas for faster processing
            console.print("[yellow]Processing large file stop_times.csv...[/yellow]")
            
            # Read in chunks to handle large file
            chunk_size = 50000
            total_rows = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Importing {table_name}", total=file_size)
                
                for chunk in pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, 
                                        chunksize=chunk_size):
                    # Convert DataFrame to list of tuples for bulk insert
                    records = chunk.to_records(index=False).tolist()
                    cursor.executemany(insert_sql, records)
                    total_rows += len(records)
                    
                    # Update progress based on file position
                    progress.update(task, advance=chunk_size * len(chunk.columns) * 50)
                    
            return total_rows
        else:
            # For smaller files, use regular csv reader
            records = []
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    values = [row.get(col, None) for col in columns]
                    records.append(values)
            
            cursor.executemany(insert_sql, records)
            return len(records)


def display_summary(stats: Dict[str, int]):
    """Display summary statistics of imported data"""
    console.print("\n[bold blue]Database Summary[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Table", style="cyan")
    table.add_column("Records", justify="right", style="green")
    
    total_records = 0
    for table_name, count in sorted(stats.items()):
        table.add_row(table_name, f"{count:,}")
        total_records += count
    
    table.add_row("─" * 20, "─" * 15, style="dim")
    table.add_row("Total", f"{total_records:,}", style="bold")
    
    console.print(table)
    
    # Additional statistics if stops table exists
    if 'stops' in stats:
        console.print("\n[bold]Geographic Coverage:[/bold]")
        # This would need a database connection to query, but we'll add it later
        console.print("[dim]Run 'oepnv-topo stats' for detailed geographic statistics[/dim]")


@cli.command()
@click.option('--db-path', default='oepnv.db', help='Path to SQLite database file')
def stats(db_path: str):
    """Display detailed statistics about the GTFS data"""
    if not os.path.exists(db_path):
        console.print(f"[bold red]Database {db_path} not found. Run 'init' command first.[/bold red]")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Basic statistics
        console.print("\n[bold blue]Database Statistics[/bold blue]")
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        # Count records in each table
        tables = ['stops', 'routes', 'trips', 'stop_times', 'transfers']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats_table.add_row(f"{table.capitalize()} count", f"{count:,}")
        
        console.print(stats_table)
        
        # Geographic bounds
        cursor.execute("""
            SELECT MIN(stop_lat), MAX(stop_lat), MIN(stop_lon), MAX(stop_lon)
            FROM stops
        """)
        min_lat, max_lat, min_lon, max_lon = cursor.fetchone()
        
        console.print("\n[bold]Geographic Bounds:[/bold]")
        console.print(f"Latitude:  {min_lat:.4f} to {max_lat:.4f}")
        console.print(f"Longitude: {min_lon:.4f} to {max_lon:.4f}")
        
        # Center point (approximate)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        console.print(f"Center:    {center_lat:.4f}, {center_lon:.4f}")
        
        # Route types
        console.print("\n[bold]Route Types:[/bold]")
        cursor.execute("""
            SELECT route_type, COUNT(*) as count
            FROM routes
            GROUP BY route_type
            ORDER BY count DESC
        """)
        
        route_type_names = {
            0: "Tram/Light Rail",
            1: "Subway/Metro",
            2: "Rail",
            3: "Bus",
            4: "Ferry",
            5: "Cable Car",
            6: "Gondola",
            7: "Funicular"
        }
        
        for route_type, count in cursor.fetchall():
            type_name = route_type_names.get(route_type, f"Type {route_type}")
            console.print(f"  {type_name}: {count:,}")
        
        # Top stops by connections
        console.print("\n[bold]Top 10 Stops by Connections:[/bold]")
        cursor.execute("""
            SELECT s.stop_name, COUNT(DISTINCT st.trip_id) as connections
            FROM stops s
            JOIN stop_times st ON s.stop_id = st.stop_id
            GROUP BY s.stop_id, s.stop_name
            ORDER BY connections DESC
            LIMIT 10
        """)
        
        for i, (stop_name, connections) in enumerate(cursor.fetchall(), 1):
            console.print(f"  {i:2d}. {stop_name}: {connections:,} connections")
        
        # Example spatial query
        console.print("\n[bold]Example: Nearest stops to Düsseldorf Hauptbahnhof:[/bold]")
        # Approximate coordinates for Düsseldorf Hbf
        example_lat, example_lon = 51.219960, 6.794160
        
        nearest = find_nearest_stops(conn, example_lat, example_lon, max_distance_km=2.0, limit=5)
        for stop in nearest:
            console.print(f"  {stop['stop_name']}: {stop['distance_km']:.2f} km")
        
    finally:
        conn.close()


@cli.command()
@click.option('--db-path', default='oepnv.db', help='Path to SQLite database file')
@click.option('--lat', type=float, required=True, help='Starting latitude')
@click.option('--lon', type=float, required=True, help='Starting longitude')
@click.option('--time', 'max_time', default=30, help='Maximum travel time in minutes')
@click.option('--max-walk', default=500, help='Maximum walking distance between stops in meters')
def build_graph(db_path: str, lat: float, lon: float, max_time: int, max_walk: int):
    """Build optimized transit network graph for a specific starting location"""
    
    # Check if database exists
    if not os.path.exists(db_path):
        console.print(f"[bold red]Database {db_path} not found. Run 'init' command first.[/bold red]")
        return
    
    console.print(f"[bold blue]Building optimized transit network graph[/bold blue]")
    
    # Connect to database and build graph
    conn = sqlite3.connect(db_path)
    
    try:
        graph = build_transit_graph(
            conn, 
            start_lat=lat, 
            start_lon=lon,
            max_time_minutes=max_time,
            max_walking_distance_m=max_walk
        )
        
        # Save with location-specific filename
        graph_file = f"transit_graph_{lat:.4f}_{lon:.4f}_{max_time}min.gpickle"
        save_graph(graph, graph_file)
        
        console.print(f"\n[bold green]Graph building completed![/bold green]")
        console.print(f"[dim]Graph saved to {graph_file}[/dim]")
        
    finally:
        conn.close()


@cli.command()
@click.option('--db-path', default='oepnv.db', help='Path to SQLite database file')
@click.option('--address', help='Address to start from')
@click.option('--lat', type=float, help='Starting latitude (alternative to address)')
@click.option('--lon', type=float, help='Starting longitude (alternative to address)')
@click.option('--time', 'max_time', default=30, help='Maximum travel time in minutes')
@click.option('--departure', default='09:00', help='Departure time (HH:MM format)')
@click.option('--visualize', is_flag=True, help='Generate interactive map after calculation')
@click.option('--map-output', default='isochrone_map.html', help='Output HTML file for map (if --visualize)')
def query(db_path: str, address: str, lat: float, lon: float, max_time: int, departure: str, visualize: bool, map_output: str):
    """Find all stops reachable within the given time from an address or coordinates"""
    
    # Check if database exists
    if not os.path.exists(db_path):
        console.print(f"[bold red]Database {db_path} not found. Run 'init' command first.[/bold red]")
        return
    
    # Determine starting coordinates
    start_lat, start_lon = None, None
    
    if address:
        console.print(f"[cyan]Geocoding address: {address}[/cyan]")
        coords = geocode_address(address)
        if coords:
            start_lat, start_lon = coords
        else:
            console.print("[red]Could not geocode address. Please provide coordinates instead.[/red]")
            return
    elif lat is not None and lon is not None:
        start_lat, start_lon = lat, lon
        console.print(f"[cyan]Using coordinates: {start_lat:.6f}, {start_lon:.6f}[/cyan]")
    else:
        console.print("[red]Please provide either an address or coordinates (--lat and --lon)[/red]")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Find all stops within walking distance (20 minutes max)
        console.print(f"\n[cyan]Finding stops within 20-minute walk of starting point...[/cyan]")
        walkable_stops = find_walkable_stops(conn, start_lat, start_lon, max_walk_time_minutes=20)
        
        if not walkable_stops:
            console.print("[yellow]No stops found within 20-minute walk of starting location[/yellow]")
            return
        
        console.print(f"[green]Found {len(walkable_stops)} stops within walking distance[/green]")
        console.print(f"[dim]Closest: {walkable_stops[0]['stop_name']} ({walkable_stops[0]['walk_time_minutes']:.1f}min walk)[/dim]")
        
        # Optimize walkable stops using line coverage
        console.print(f"\n[cyan]Optimizing origins using line coverage analysis...[/cyan]")
        stop_lines_mapping = get_stop_lines_mapping(conn)
        optimized_walkable_stops = optimize_walkable_stops_by_line_coverage(walkable_stops, stop_lines_mapping)
        
        # Build optimized graph that includes all potentially reachable stops
        console.print(f"\n[cyan]Building transit graph for {max_time}-minute isochrone...[/cyan]")
        
        # Use the closest optimized stop as the center for graph filtering
        center_stop = optimized_walkable_stops[0] if optimized_walkable_stops else walkable_stops[0]
        graph = build_transit_graph(
            conn, 
            start_lat=center_stop['stop_lat'], 
            start_lon=center_stop['stop_lon'],
            max_time_minutes=max_time,
            max_walking_distance_m=500
        )
        
        if graph.number_of_nodes() == 0:
            console.print("[yellow]No reachable stops found for the given parameters[/yellow]")
            return
        
        # Calculate multi-origin isochrone
        console.print(f"\n[cyan]Calculating isochrone with enhanced walking model...[/cyan]")
        console.print(f"[dim]Maximum walking time: 20 minutes at start and end of journey[/dim]")
        
        reachable_stops = calculate_multi_origin_isochrone(graph, optimized_walkable_stops, max_time)
        
        if not reachable_stops:
            console.print("[yellow]No stops reachable within time limit[/yellow]")
            return
        
        console.print(f"\n[bold green]Found {len(reachable_stops)} reachable stops within {max_time} minutes![/bold green]")
        
        # Add end-of-journey walking expansion
        expanded_points = add_end_walking_expansion(reachable_stops, graph, max_walk_time_minutes=20, max_total_time_minutes=max_time)
        
        console.print(f"[bold green]Total reachable area: {len(expanded_points)} points within {max_time} minutes![/bold green]")
        console.print(f"[dim]Including 20-minute walking at start and end of journey[/dim]")
        
        # Group expanded points by total travel time ranges  
        time_ranges = [
            (0, 15, "0-15 min"),
            (15, 30, "15-30 min"), 
            (30, 45, "30-45 min"),
            (45, 60, "45-60 min"),
            (60, 999, "60+ min")
        ]
        
        for min_time, max_range, label in time_ranges:
            if max_range > max_time:
                max_range = max_time
            if min_time >= max_time:
                break
                
            points_in_range = [
                (point_id, info) for point_id, info in expanded_points.items()
                if min_time <= info['total_time_minutes'] <= max_range
            ]
            
            if points_in_range:
                transit_stops = [p for p in points_in_range if p[1]['point_type'] == 'transit_stop']
                walking_points = [p for p in points_in_range if p[1]['point_type'] == 'walking_destination']
                
                console.print(f"\n[bold]{label}: {len(points_in_range)} total points[/bold]")
                console.print(f"[dim]  {len(transit_stops)} transit stops + {len(walking_points)} walking destinations[/dim]")
                
                # Show a few examples with breakdown
                examples = sorted(points_in_range, key=lambda x: x[1]['total_time_minutes'])[:3]
                for point_id, info in examples:
                    if info['point_type'] == 'transit_stop':
                        console.print(f"  🚏 {info['name'][:30]}: {info['total_time_minutes']:.1f}min " +
                                    f"(walk {info.get('walk_time_minutes', 0):.1f} + transit {info.get('transit_time_minutes', 0):.1f})")
                    else:
                        console.print(f"  🚶 {info['name'][:30]}: {info['total_time_minutes']:.1f}min " +
                                    f"(+{info['end_walk_time_minutes']:.1f}min walk from transit)")
                
                if len(points_in_range) > 3:
                    console.print(f"  ... and {len(points_in_range) - 3} more points")
                    
        # Show summary statistics
        total_transit_stops = len([p for p in expanded_points.values() if p['point_type'] == 'transit_stop'])
        total_walking_dest = len([p for p in expanded_points.values() if p['point_type'] == 'walking_destination'])
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Transit stops reachable: {total_transit_stops}")
        console.print(f"  Additional walking destinations: {total_walking_dest}")
        console.print(f"  Total reachable area: {len(expanded_points)} points")
        
        # Generate visualization if requested
        if visualize:
            console.print(f"\n[cyan]Generating interactive map visualization...[/cyan]")
            
            # Convert to format expected by visualization functions
            visualization_data = {}
            for point_id, info in expanded_points.items():
                # Handle different data structures for transit stops vs walking destinations
                if info.get('point_type') == 'transit_stop':
                    lat = info.get('lat') or info.get('stop_lat')
                    lon = info.get('lon') or info.get('stop_lon')
                    name = info.get('name') or info.get('stop_name', f'Stop {point_id}')
                else:
                    lat = info.get('lat')
                    lon = info.get('lon')
                    name = info.get('name', f'Walking point {point_id}')
                
                # Only include points with valid coordinates
                if lat is not None and lon is not None:
                    visualization_data[point_id] = {
                        'stop_lat': lat,
                        'stop_lon': lon, 
                        'stop_name': name,
                        'travel_time': info.get('total_time_minutes', 0),
                        'type': 'transit' if info.get('point_type') == 'transit_stop' else 'walking'
                    }
            
            # Generate map title
            title = f"Reachable Area within {max_time} minutes"
            if address:
                title = f"{title} from {address}"
            
            # Create the interactive map
            map_path = create_isochrone_map(
                start_lat, start_lon, visualization_data, title, map_output
            )
            
            if map_path:
                console.print(f"\n[bold green]✓ Interactive map generated![/bold green]")
                console.print(f"[green]Open in browser: file://{map_path}[/green]")
            else:
                console.print("[red]Failed to generate map[/red]")
        
        console.print(f"\n[dim]Data © OpenStreetMap contributors[/dim]")
        
    finally:
        conn.close()


@cli.command()
@click.option('--db-path', default='oepnv.db', help='Path to SQLite database file')
@click.option('--route', help='Route number to search for (e.g., "447")')
@click.option('--stop', help='Stop name to search for')
@click.option('--address1', help='First address to test connection')
@click.option('--address2', help='Second address to test connection')
def debug(db_path: str, route: str, stop: str, address1: str, address2: str):
    """Debug and investigate route connectivity issues"""
    
    if not os.path.exists(db_path):
        console.print(f"[bold red]Database {db_path} not found. Run 'init' command first.[/bold red]")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        if route:
            console.print(f"[cyan]Searching for route/line: {route}[/cyan]")
            
            # Search in routes table
            cursor.execute("""
                SELECT route_id, route_short_name, route_long_name, route_type 
                FROM routes 
                WHERE route_short_name LIKE ? OR route_long_name LIKE ? OR route_id LIKE ?
            """, (f'%{route}%', f'%{route}%', f'%{route}%'))
            
            routes = cursor.fetchall()
            if routes:
                console.print(f"[green]Found {len(routes)} matching routes:[/green]")
                for route_id, short_name, long_name, route_type in routes:
                    console.print(f"  • {short_name or 'N/A'} - {long_name or 'N/A'} (ID: {route_id}, Type: {route_type})")
                
                # For each route, find stops
                for route_id, short_name, long_name, route_type in routes[:3]:  # Limit to first 3
                    console.print(f"\n[cyan]Stops for route {short_name or route_id}:[/cyan]")
                    cursor.execute("""
                        SELECT DISTINCT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
                        FROM stops s
                        JOIN stop_times st ON s.stop_id = st.stop_id
                        JOIN trips t ON st.trip_id = t.trip_id
                        WHERE t.route_id = ?
                        ORDER BY s.stop_name
                        LIMIT 10
                    """, (route_id,))
                    
                    route_stops = cursor.fetchall()
                    for stop_id, stop_name, lat, lon in route_stops:
                        console.print(f"    {stop_name} ({lat:.4f}, {lon:.4f})")
                    
                    if len(route_stops) > 10:
                        console.print(f"    ... and {len(route_stops) - 10} more stops")
            else:
                console.print(f"[yellow]No routes found matching '{route}'[/yellow]")
        
        if stop:
            console.print(f"\n[cyan]Searching for stop: {stop}[/cyan]")
            cursor.execute("""
                SELECT stop_id, stop_name, stop_lat, stop_lon 
                FROM stops 
                WHERE stop_name LIKE ?
                ORDER BY stop_name
                LIMIT 10
            """, (f'%{stop}%',))
            
            stops = cursor.fetchall()
            if stops:
                console.print(f"[green]Found {len(stops)} matching stops:[/green]")
                for stop_id, stop_name, lat, lon in stops:
                    console.print(f"  • {stop_name} ({lat:.4f}, {lon:.4f}) - ID: {stop_id}")
            else:
                console.print(f"[yellow]No stops found matching '{stop}'[/yellow]")
        
        if address1 and address2:
            console.print(f"\n[cyan]Testing connection: {address1} → {address2}[/cyan]")
            
            # Geocode both addresses
            coords1 = geocode_address(address1)
            coords2 = geocode_address(address2)
            
            if coords1 and coords2:
                lat1, lon1 = coords1
                lat2, lon2 = coords2
                
                # Find closest stops to each address
                console.print(f"\n[cyan]Finding closest stops to each address...[/cyan]")
                
                def find_closest_stops(lat, lon, limit=5):
                    cursor.execute("""
                        SELECT stop_id, stop_name, stop_lat, stop_lon,
                               (({} - stop_lat) * ({} - stop_lat) + ({} - stop_lon) * ({} - stop_lon)) as dist_sq
                        FROM stops
                        ORDER BY dist_sq
                        LIMIT {}
                    """.format(lat, lat, lon, lon, limit))
                    return cursor.fetchall()
                
                stops1 = find_closest_stops(lat1, lon1)
                stops2 = find_closest_stops(lat2, lon2)
                
                console.print(f"[green]Closest stops to {address1}:[/green]")
                for stop_id, name, lat, lon, dist_sq in stops1:
                    distance_km = haversine_distance(lat1, lon1, lat, lon)
                    console.print(f"  • {name} - {distance_km:.2f}km away")
                
                console.print(f"\n[green]Closest stops to {address2}:[/green]")
                for stop_id, name, lat, lon, dist_sq in stops2:
                    distance_km = haversine_distance(lat2, lon2, lat, lon)
                    console.print(f"  • {name} - {distance_km:.2f}km away")
            else:
                console.print("[red]Could not geocode one or both addresses[/red]")
    
    finally:
        conn.close()


@cli.command()
@click.option('--db-path', default='oepnv.db', help='Path to SQLite database file')
@click.option('--address', help='Address to start from')
@click.option('--lat', type=float, help='Starting latitude (alternative to address)')
@click.option('--lon', type=float, help='Starting longitude (alternative to address)')
@click.option('--time', 'max_time', default=30, help='Maximum travel time in minutes')
@click.option('--output', default='isochrone_map.html', help='Output HTML file name')
@click.option('--simple', is_flag=True, help='Create simple boundary map instead of time-layered')
def visualize(db_path: str, address: str, lat: float, lon: float, max_time: int, output: str, simple: bool):
    """Generate interactive map visualization of reachable areas"""
    
    # Check if database exists
    if not os.path.exists(db_path):
        console.print(f"[bold red]Database {db_path} not found. Run 'init' command first.[/bold red]")
        return
    
    # Determine starting coordinates
    start_lat, start_lon = None, None
    
    if address:
        console.print(f"[cyan]Geocoding address: {address}[/cyan]")
        coords = geocode_address(address)
        if coords:
            start_lat, start_lon = coords
        else:
            console.print("[red]Could not geocode address. Please provide coordinates instead.[/red]")
            return
    elif lat is not None and lon is not None:
        start_lat, start_lon = lat, lon
        console.print(f"[cyan]Using coordinates: {start_lat:.6f}, {start_lon:.6f}[/cyan]")
    else:
        console.print("[red]Please provide either --address or both --lat and --lon[/red]")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Calculate isochrone data (reuse logic from query command)
        console.print(f"\n[cyan]Calculating {max_time}-minute isochrone for visualization...[/cyan]")
        
        # Find walkable stops
        walkable_stops = find_walkable_stops(conn, start_lat, start_lon, max_walk_time_minutes=20)
        
        if not walkable_stops:
            console.print("[yellow]No stops found within 20-minute walk of starting location[/yellow]")
            return
        
        # Optimize walkable stops using line coverage
        stop_lines_mapping = get_stop_lines_mapping(conn)
        optimized_walkable_stops = optimize_walkable_stops_by_line_coverage(walkable_stops, stop_lines_mapping)
        
        # Build transit graph
        center_stop = optimized_walkable_stops[0] if optimized_walkable_stops else walkable_stops[0]
        graph = build_transit_graph(
            conn, 
            start_lat=center_stop['stop_lat'], 
            start_lon=center_stop['stop_lon'],
            max_time_minutes=max_time,
            max_walking_distance_m=500
        )
        
        if graph.number_of_nodes() == 0:
            console.print("[yellow]No reachable stops found for the given parameters[/yellow]")
            return
        
        # Calculate multi-origin isochrone
        reachable_stops = calculate_multi_origin_isochrone(graph, optimized_walkable_stops, max_time)
        
        if not reachable_stops:
            console.print("[yellow]No stops reachable within time limit[/yellow]")
            return
        
        # Add end-of-journey walking expansion
        expanded_points = add_end_walking_expansion(reachable_stops, graph, max_walk_time_minutes=20, max_total_time_minutes=max_time)
        
        # Convert to format expected by visualization functions
        visualization_data = {}
        for point_id, info in expanded_points.items():
            # Handle different data structures for transit stops vs walking destinations
            if info.get('point_type') == 'transit_stop':
                lat = info.get('lat') or info.get('stop_lat')
                lon = info.get('lon') or info.get('stop_lon')
                name = info.get('name') or info.get('stop_name', f'Stop {point_id}')
            else:
                lat = info.get('lat')
                lon = info.get('lon')
                name = info.get('name', f'Walking point {point_id}')
            
            # Only include points with valid coordinates
            if lat is not None and lon is not None:
                visualization_data[point_id] = {
                    'stop_lat': lat,
                    'stop_lon': lon, 
                    'stop_name': name,
                    'travel_time': info.get('total_time_minutes', 0),
                    'type': 'transit' if info.get('point_type') == 'transit_stop' else 'walking'
                }
        
        # Generate map
        title = f"Reachable Area within {max_time} minutes"
        if address:
            title = f"{title} from {address}"
        
        if simple:
            map_path = create_simple_boundary_map(
                start_lat, start_lon, visualization_data, max_time, title, output
            )
        else:
            map_path = create_isochrone_map(
                start_lat, start_lon, visualization_data, title, output
            )
        
        if map_path:
            console.print(f"\n[bold green]✓ Interactive map generated successfully![/bold green]")
            console.print(f"[green]Open in browser: file://{map_path}[/green]")
        else:
            console.print("[red]Failed to generate map[/red]")
        
    finally:
        conn.close()


if __name__ == '__main__':
    cli()