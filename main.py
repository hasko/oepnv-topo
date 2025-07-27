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


console = Console()


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


if __name__ == '__main__':
    cli()