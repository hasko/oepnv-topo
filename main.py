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
def query(db_path: str, address: str, lat: float, lon: float, max_time: int, departure: str):
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
        # Build optimized graph on-demand
        console.print(f"\n[cyan]Building transit graph for {max_time}-minute isochrone...[/cyan]")
        
        graph = build_transit_graph(
            conn, 
            start_lat=start_lat, 
            start_lon=start_lon,
            max_time_minutes=max_time,
            max_walking_distance_m=500
        )
        
        if graph.number_of_nodes() == 0:
            console.print("[yellow]No reachable stops found for the given parameters[/yellow]")
            return
        
        # Find the closest stop to starting point as the actual start node
        nearest_stops = find_nearest_stops(conn, start_lat, start_lon, max_distance_km=2.0, limit=5)
        
        if not nearest_stops:
            console.print("[yellow]No nearby stops found within 2km to start journey[/yellow]")
            return
        
        start_stop_id = nearest_stops[0]['stop_id']
        console.print(f"[cyan]Starting from: {nearest_stops[0]['stop_name']} (walk {nearest_stops[0]['distance_km']:.2f}km)[/cyan]")
        
        # Calculate reachable stops using Dijkstra's algorithm
        console.print(f"[cyan]Calculating reachable stops within {max_time} minutes...[/cyan]")
        
        # Convert max_time to seconds and add walking time to start
        max_time_seconds = max_time * 60
        walk_to_start_seconds = (nearest_stops[0]['distance_km'] * 1000) / 1.4  # 5 km/h walking
        available_transit_time = max_time_seconds - walk_to_start_seconds
        
        if available_transit_time <= 0:
            console.print("[yellow]Starting location is too far to walk to nearest stop within time limit[/yellow]")
            return
        
        # Run Dijkstra's algorithm
        if graph.has_node(start_stop_id):
            reachable = nx.single_source_dijkstra_path_length(
                graph, start_stop_id, cutoff=available_transit_time, weight='weight'
            )
            
            console.print(f"\n[bold green]Found {len(reachable)} reachable stops within {max_time} minutes![/bold green]")
            
            # Group stops by travel time ranges
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
                    
                stops_in_range = [
                    (stop_id, time_sec) for stop_id, time_sec in reachable.items()
                    if min_time * 60 <= time_sec <= max_range * 60
                ]
                
                if stops_in_range:
                    console.print(f"\n[bold]{label}: {len(stops_in_range)} stops[/bold]")
                    
                    # Show a few examples
                    for stop_id, time_sec in sorted(stops_in_range, key=lambda x: x[1])[:5]:
                        stop_name = graph.nodes[stop_id]['name']
                        console.print(f"  {stop_name[:40]}: {time_sec/60:.1f} min")
                    
                    if len(stops_in_range) > 5:
                        console.print(f"  ... and {len(stops_in_range) - 5} more")
            
        else:
            console.print(f"[yellow]Starting stop {start_stop_id} not found in graph[/yellow]")
        
        console.print(f"\n[dim]Data © OpenStreetMap contributors[/dim]")
        
    finally:
        conn.close()


if __name__ == '__main__':
    cli()