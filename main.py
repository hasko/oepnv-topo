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
import math
from math import radians, cos, sin, acos, sqrt, atan2
from typing import Dict, Tuple, List, Optional, Union
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import json
from datetime import datetime, timedelta
# networkx removed - using database-driven approach
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union, transform
import pyproj
from functools import partial
from visualization import create_isochrone_map, create_simple_boundary_map, create_circle_union_map


console = Console()


def safe_difference(geom1, geom2):
    """
    Safely compute geometric difference with validation and error handling
    
    Args:
        geom1: Primary geometry (Shapely geometry)
        geom2: Geometry to subtract (Shapely geometry)
    
    Returns:
        Result of geom1.difference(geom2) or None if operation fails/empty
    """
    try:
        # Handle None inputs
        if geom1 is None or geom2 is None:
            return geom1
        
        # Ensure input geometries are valid
        if hasattr(geom1, 'is_valid') and not geom1.is_valid:
            geom1 = geom1.buffer(0)
        if hasattr(geom2, 'is_valid') and not geom2.is_valid:
            geom2 = geom2.buffer(0)
            
        # Compute difference
        result = geom1.difference(geom2)
        
        # Validate and clean result
        if hasattr(result, 'is_valid') and not result.is_valid:
            result = result.buffer(0)
            
        # Handle empty results
        if hasattr(result, 'is_empty') and result.is_empty:
            return None
            
        return result
        
    except Exception as e:
        console.print(f"[yellow]Warning: Geometric difference operation failed: {e}[/yellow]")
        return geom1  # Fallback to original geometry


def safe_union(geometries):
    """
    Safely compute union of multiple geometries with error handling
    
    Args:
        geometries: List of Shapely geometries
    
    Returns:
        Union geometry or None if operation fails
    """
    try:
        if not geometries:
            return None
            
        # Filter out None geometries and validate
        valid_geometries = []
        for geom in geometries:
            if geom is not None:
                if hasattr(geom, 'is_valid') and not geom.is_valid:
                    geom = geom.buffer(0)
                valid_geometries.append(geom)
        
        if not valid_geometries:
            return None
            
        # Compute union
        result = unary_union(valid_geometries)
        
        # Validate result
        if hasattr(result, 'is_valid') and not result.is_valid:
            result = result.buffer(0)
            
        return result if not (hasattr(result, 'is_empty') and result.is_empty) else None
        
    except Exception as e:
        console.print(f"[yellow]Warning: Geometric union operation failed: {e}[/yellow]")
        return None

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


def is_service_active_on_date(conn: sqlite3.Connection, service_id: str, date_str: str) -> bool:
    """
    Check if a service is active on a specific date based on calendar and calendar_dates
    
    Args:
        conn: Database connection
        service_id: Service ID to check
        date_str: Date in YYYYMMDD format (e.g., '20250407')
    
    Returns:
        True if service is active on this date
    """
    cursor = conn.cursor()
    
    # First check calendar_dates for exceptions
    cursor.execute("""
        SELECT exception_type 
        FROM calendar_dates 
        WHERE service_id = ? AND date = ?
    """, (service_id, date_str))
    
    exception = cursor.fetchone()
    if exception:
        # exception_type: 1 = service added, 2 = service removed
        return exception[0] == 1
    
    # No exception found, check regular calendar
    # Parse date to get day of week (0 = Monday, 6 = Sunday)
    from datetime import datetime
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    weekday = date_obj.weekday()
    
    # Map weekday to column names
    day_columns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    day_column = day_columns[weekday]
    
    # Check if service runs on this day of week and date is within service period
    query = f"""
        SELECT {day_column}
        FROM calendar
        WHERE service_id = ? 
          AND start_date <= ?
          AND end_date >= ?
    """
    
    cursor.execute(query, (service_id, date_str, date_str))
    result = cursor.fetchone()
    
    return bool(result and result[0] == 1)


def get_active_service_ids_for_date(conn: sqlite3.Connection, date_str: str) -> List[str]:
    """
    Get all service IDs that are active on a specific date.
    Combines both calendar (regular schedule) and calendar_dates (exceptions) logic.
    
    Args:
        conn: Database connection
        date_str: Date in YYYYMMDD format (e.g., '20250730')
    
    Returns:
        List of active service_ids for the given date
    """
    cursor = conn.cursor()
    
    # First, get all service_ids that exist
    cursor.execute("SELECT DISTINCT service_id FROM trips")
    all_service_ids = [row[0] for row in cursor.fetchall()]
    
    active_service_ids = []
    
    for service_id in all_service_ids:
        if is_service_active_on_date(conn, service_id, date_str):
            active_service_ids.append(service_id)
    
    return active_service_ids


def get_active_trips_on_date(conn: sqlite3.Connection, date_str: str) -> List[str]:
    """
    Get all trip IDs that are active on a specific date
    
    Args:
        date_str: Date in YYYYMMDD format (e.g., '20250407')
    
    Returns:
        List of active trip IDs
    """
    cursor = conn.cursor()
    
    # Get all trips with their service IDs
    cursor.execute("SELECT trip_id, service_id FROM trips")
    
    active_trips = []
    for trip_id, service_id in cursor.fetchall():
        if is_service_active_on_date(conn, service_id, date_str):
            active_trips.append(trip_id)
    
    return active_trips


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
    cursor.execute("""
        SELECT stop_id, stop_name, stop_lat, stop_lon 
        FROM stops
    """)
    
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


def get_route_mappings(conn: sqlite3.Connection) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Create robust route mappings using GTFS unique identifiers
    
    Returns:
        - route_name_to_id: {route_short_name -> route_id}  
        - route_id_to_name: {route_id -> route_short_name}
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT route_id, route_short_name, route_long_name
        FROM routes 
        WHERE route_short_name IS NOT NULL AND route_short_name != ''
    """)
    
    name_to_id = {}
    id_to_name = {}
    
    for route_id, short_name, long_name in cursor.fetchall():
        # Primary mapping: short_name -> route_id
        name_to_id[short_name] = route_id
        id_to_name[route_id] = short_name
        
        # Secondary mapping: long_name -> route_id (if different)
        if long_name and long_name != short_name:
            name_to_id[long_name] = route_id
    
    return name_to_id, id_to_name


def get_stop_lines_mapping(conn: sqlite3.Connection) -> Dict[str, set]:
    """
    Create mapping of stop_id -> set of human-readable route short names (like "447", "U43", "X13")
    This is used for line coverage optimization and display
    Uses route_id joins for robustness, route_short_name only for display
    """
    cursor = conn.cursor()
    
    # Query using route_id (unique key) joins, route_short_name only for display
    query = """
    SELECT DISTINCT st.stop_id, r.route_short_name as line_name
    FROM stop_times st
    JOIN trips t ON st.trip_id = t.trip_id
    JOIN routes r ON t.route_id = r.route_id
    WHERE st.stop_id IS NOT NULL 
      AND t.route_id IS NOT NULL
      AND r.route_short_name IS NOT NULL
      AND r.route_short_name != ''
    """
    
    stop_lines = {}
    cursor.execute(query)
    
    for stop_id, line_name in cursor.fetchall():
        if stop_id not in stop_lines:
            stop_lines[stop_id] = set()
        stop_lines[stop_id].add(line_name)
    
    return stop_lines


def suggest_alternative_dates_for_route(conn: sqlite3.Connection, route_name: str, 
                                       target_date: str, days_range: int = 7) -> List[str]:
    """
    When a route isn't running on target date, suggest nearby dates when it is active
    
    Args:
        route_name: Route short name (e.g., "S1")
        target_date: Date in YYYYMMDD format
        days_range: How many days before/after to check
    
    Returns:
        List of dates in YYYYMMDD format when route is active
    """
    cursor = conn.cursor()
    
    # Get route_id from name
    route_mappings, _ = get_route_mappings(conn)
    route_id = route_mappings.get(route_name)
    
    if not route_id:
        return []
    
    # Convert target date to check nearby dates
    from datetime import datetime, timedelta
    try:
        target_dt = datetime.strptime(target_date, '%Y%m%d')
    except ValueError:
        return []
    
    alternative_dates = []
    
    # Check days before and after target date
    for days_offset in range(-days_range, days_range + 1):
        if days_offset == 0:
            continue  # Skip target date (we know it doesn't work)
            
        check_date = target_dt + timedelta(days=days_offset)
        check_date_str = check_date.strftime('%Y%m%d')
        
        # Check if route is active on this date using proper calendar validation
        active_service_ids = get_active_service_ids_for_date(conn, check_date_str)
        if not active_service_ids:
            continue  # No services active on this date
        
        # Convert service_ids to SQL IN clause
        service_placeholders = ','.join(['?'] * len(active_service_ids))
        cursor.execute(f"""
            SELECT COUNT(*) as active_trips
            FROM trips t 
            WHERE t.route_id = ?
              AND t.service_id IN ({service_placeholders})
        """, [route_id] + active_service_ids)
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            alternative_dates.append(check_date_str)
    
    return alternative_dates


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
    
    reduction_percentage = ((len(walkable_stops) - len(selected_stops)) / len(walkable_stops)) * 100
    console.print(f"[green]Line coverage optimization: {len(walkable_stops)} → {len(selected_stops)} stops[/green]")
    console.print(f"[dim]Reduction: {len(walkable_stops) - len(selected_stops)} stops eliminated ({reduction_percentage:.1f}% fewer origins)[/dim]")
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
    
    console.print(f"[green]✓ Optimized origins: Using {len(selected_stops)} stops instead of {len(walkable_stops)} for routing[/green]")
    
    return selected_stops


def seconds_to_time_str(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM:SS format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def time_str_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to seconds since midnight, handling times >= 24:00:00"""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1]) 
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def database_line_following_isochrone(conn: sqlite3.Connection, 
                                    walkable_stops: List[Dict],
                                    start_time_seconds: int,
                                    max_total_time_minutes: int,
                                    date_str: str,
                                    verbose: bool = False) -> Dict[str, Dict]:
    """
    Database-driven line-following algorithm that explores transit lines and transfers
    without building a graph in memory. Uses targeted SQL queries to follow lines.
    
    Args:
        conn: Database connection
        walkable_stops: List of stops reachable by walking from origin
        start_time_seconds: Departure time in seconds since midnight  
        max_total_time_minutes: Maximum total travel time
        date_str: Date in YYYYMMDD format for service validation
        
    Returns:
        Dictionary mapping stop_id to travel information
    """
    max_total_time_seconds = max_total_time_minutes * 60
    end_time_seconds = start_time_seconds + max_total_time_seconds
    
    console.print(f"[cyan]Database line-following from {len(walkable_stops)} walkable stops...[/cyan]")
    console.print(f"[dim]Time window: {seconds_to_time_str(start_time_seconds)} - {seconds_to_time_str(end_time_seconds)}[/dim]")
    console.print(f"[dim]Date: {date_str}[/dim]")
    
    # Track visited stops with earliest arrival time
    visited = {}  # stop_id -> earliest_arrival_time_seconds
    
    # Simple queue for exploration: (stop_id, arrival_time, total_time_minutes)
    to_explore = []
    
    # Initialize with walkable starting stops
    for walkable_stop in walkable_stops:
        stop_id = walkable_stop['stop_id']
        walk_time_seconds = int(walkable_stop['walk_time_minutes'] * 60)
        arrival_at_stop = start_time_seconds + walk_time_seconds
        
        if arrival_at_stop < end_time_seconds:
            to_explore.append((stop_id, arrival_at_stop, walk_time_seconds / 60.0))
    
    console.print(f"[green]Starting with {len(to_explore)} walkable stops[/green]")
    
    # Result storage
    reachable_stops = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Exploring transit network...", total=None)
        
        iterations = 0
        while to_explore:
            iterations += 1
            
            if iterations % 50 == 0:
                progress.update(task, description=f"Exploring... ({len(visited)} visited, {len(to_explore)} queued)")
            
            # Get next stop to explore
            current_stop, arrival_time, total_time_minutes = to_explore.pop(0)
            
            # Skip if we've already visited this stop with a better time
            if current_stop in visited and visited[current_stop] <= arrival_time:
                if verbose:
                    console.print(f"[dim]~~{current_stop}~~ (visited earlier at {seconds_to_time_str(visited[current_stop])})[/dim]")
                continue
            
            # Mark as visited with this arrival time
            visited[current_stop] = arrival_time
            
            if verbose:
                # Get human-readable stop name
                cursor = conn.cursor()
                cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (current_stop,))
                result = cursor.fetchone()
                stop_name = result[0] if result else current_stop
                remaining_time = max_total_time_minutes - total_time_minutes
                console.print(f"[cyan]🚏 {stop_name} (arrive {seconds_to_time_str(arrival_time)}, {remaining_time:.1f}min budget)[/cyan]")
            
            # Add to results
            reachable_stops[current_stop] = {
                'arrival_time': arrival_time,
                'total_time_minutes': total_time_minutes
            }
            
            # Find all trips departing from this stop after we arrive
            departing_trips = query_departing_trips_simple(
                conn, current_stop, arrival_time, end_time_seconds, date_str
            )
            
            if verbose and departing_trips:
                console.print(f"  Found {len(departing_trips)} departing trips:")
            
            for trip_id, departure_time, boarding_sequence, route_name in departing_trips:
                if verbose:
                    wait_time = (departure_time - arrival_time) / 60.0
                    console.print(f"    📍 {route_name} departs {seconds_to_time_str(departure_time)} (wait {wait_time:.1f}min)")
                
                # Get ALL downstream stops on this trip in one query
                downstream_stops = query_all_stops_on_trip(
                    conn, trip_id, boarding_sequence, end_time_seconds
                )
                
                if verbose:
                    console.print(f"      → {len(downstream_stops)} stops on this trip")
                
                # Add each reachable stop to exploration queue
                for stop_id, stop_arrival_time in downstream_stops:
                    # Calculate total travel time
                    total_time_to_stop = (stop_arrival_time - start_time_seconds) / 60.0
                    
                    # Only explore if we haven't seen it with better time
                    if stop_id not in visited or stop_arrival_time < visited[stop_id]:
                        to_explore.append((stop_id, stop_arrival_time, total_time_to_stop))
                        if verbose:
                            cursor = conn.cursor()
                            cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (stop_id,))
                            result = cursor.fetchone()
                            dest_name = result[0] if result else stop_id
                            console.print(f"        ✓ {dest_name} at {seconds_to_time_str(stop_arrival_time)} ({total_time_to_stop:.1f}min total)")
                    elif verbose:
                        cursor = conn.cursor()
                        cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (stop_id,))
                        result = cursor.fetchone()
                        dest_name = result[0] if result else stop_id
                        console.print(f"        ~~{dest_name}~~ (already visited faster)")
            
            # Also check transfers from this stop
            if total_time_minutes < max_total_time_minutes - 5:  # At least 5 min for transfers
                transfers = query_direct_transfers(conn, current_stop)
                
                if verbose and transfers:
                    console.print(f"  Found {len(transfers)} transfer options:")
                
                for to_stop_id, transfer_time_seconds in transfers:
                    transfer_arrival = arrival_time + (transfer_time_seconds or 300)  # Default 5 min
                    
                    if transfer_arrival < end_time_seconds:
                        total_time_to_transfer = (transfer_arrival - start_time_seconds) / 60.0
                        
                        if to_stop_id not in visited or transfer_arrival < visited[to_stop_id]:
                            to_explore.append((to_stop_id, transfer_arrival, total_time_to_transfer))
                            if verbose:
                                cursor = conn.cursor()
                                cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (to_stop_id,))
                                result = cursor.fetchone()
                                transfer_name = result[0] if result else to_stop_id
                                transfer_minutes = (transfer_time_seconds or 300) / 60.0
                                
                                # Get routes/lines available at this transfer stop
                                cursor.execute("""
                                    SELECT DISTINCT r.route_short_name as route_name
                                    FROM routes r
                                    JOIN trips t ON r.route_id = t.route_id
                                    JOIN stop_times st ON t.trip_id = st.trip_id
                                    WHERE st.stop_id = ?
                                    AND r.route_short_name IS NOT NULL
                                    AND r.route_short_name != ''
                                    ORDER BY route_name
                                    LIMIT 10
                                """, (to_stop_id,))
                                routes = [row[0] for row in cursor.fetchall()]
                                routes_str = ", ".join(routes) if routes else "no routes"
                                if len(routes) > 10:
                                    routes_str += " ..."
                                
                                console.print(f"    🔄 Transfer to {transfer_name} ({transfer_minutes:.1f}min transfer) - lines: {routes_str}")
                        elif verbose:
                            cursor = conn.cursor()
                            cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (to_stop_id,))
                            result = cursor.fetchone()
                            transfer_name = result[0] if result else to_stop_id
                            console.print(f"    ~~Transfer to {transfer_name}~~ (already visited faster)")
    
    console.print(f"[green]Explored {iterations} iterations, found {len(reachable_stops)} reachable stops[/green]")
    return reachable_stops


def query_departing_trips_simple(conn: sqlite3.Connection, stop_id: str, 
                                earliest_departure: int, latest_departure: int,
                                date_str: str) -> List[tuple]:
    """Find trips departing from a stop within time window"""
    cursor = conn.cursor()
    
    # Use numeric time comparison to handle both zero-padded (08:00:00) and single-digit (8:00:00) formats
    # First get all potential departures, then filter by time in Python for robustness
    
    # Get active service IDs for proper calendar validation
    active_service_ids = get_active_service_ids_for_date(conn, date_str)
    if not active_service_ids:
        return []  # No services active on this date
    
    # Build query with proper service filtering
    service_placeholders = ','.join(['?'] * len(active_service_ids))
    query = f"""
    SELECT DISTINCT
        st.trip_id,
        st.departure_time,
        st.stop_sequence,
        r.route_short_name
    FROM stop_times st
    JOIN trips t ON st.trip_id = t.trip_id
    JOIN routes r ON t.route_id = r.route_id
    WHERE st.stop_id = ?
      AND t.service_id IN ({service_placeholders})
      AND EXISTS (
          -- Ensure trip has at least one stop within our time window
          SELECT 1 FROM stop_times st2
          WHERE st2.trip_id = st.trip_id
      )
    ORDER BY st.departure_time
    """
    
    params = [stop_id] + active_service_ids
    cursor.execute(query, params)
    
    # Filter results by time window using numeric comparison
    results = []
    for trip_id, dep_time_str, stop_seq, route_name in cursor.fetchall():
        try:
            dep_time_seconds = time_str_to_seconds(dep_time_str)
            # Only include departures within our time window
            if earliest_departure <= dep_time_seconds <= latest_departure:
                results.append((trip_id, dep_time_seconds, stop_seq, route_name))
        except (ValueError, AttributeError):
            # Skip invalid time formats
            continue
    
    # Limit results and sort by departure time
    results.sort(key=lambda x: x[1])  # Sort by departure time (seconds)
    return results[:100]


def query_all_stops_on_trip(conn: sqlite3.Connection, trip_id: str,
                           after_sequence: int, before_time: int) -> List[tuple]:
    """Get all downstream stops on a trip"""
    cursor = conn.cursor()
    
    # Get all stops on trip after the boarding sequence, then filter by time in Python
    query = """
    SELECT stop_id, arrival_time
    FROM stop_times
    WHERE trip_id = ?
      AND stop_sequence > ?
    ORDER BY stop_sequence
    """
    
    cursor.execute(query, (trip_id, after_sequence))
    
    results = []
    for stop_id, arrival_str in cursor.fetchall():
        try:
            arrival_seconds = time_str_to_seconds(arrival_str)
            # Only include stops we can reach within our time limit
            if arrival_seconds <= before_time:
                results.append((stop_id, arrival_seconds))
        except (ValueError, AttributeError):
            # Skip invalid time formats
            continue
    
    return results


def query_direct_transfers(conn: sqlite3.Connection, from_stop_id: str) -> List[tuple]:
    """Find direct transfer opportunities from a stop"""
    cursor = conn.cursor()
    
    query = """
    SELECT to_stop_id, min_transfer_time
    FROM transfers
    WHERE from_stop_id = ?
      AND from_stop_id != to_stop_id
    LIMIT 20
    """
    
    cursor.execute(query, (from_stop_id,))
    return cursor.fetchall()


def query_departing_lines(conn: sqlite3.Connection, stop_id: str, earliest_departure: int,
                         latest_departure: int, active_trips: set) -> List[Dict]:
    """Find all lines departing from a stop within a time window"""
    cursor = conn.cursor()
    
    # Convert seconds to time strings for SQL query
    earliest_time_str = seconds_to_time_str(earliest_departure)
    latest_time_str = seconds_to_time_str(latest_departure)
    
    # Create a temporary table for active trips to avoid large IN clauses
    # Use a more compatible approach that works with older SQLite versions
    if len(active_trips) > 1000:
        # For large trip sets, use a different approach
        active_trips_sample = list(active_trips)[:1000]  # Limit to avoid query complexity
    else:
        active_trips_sample = list(active_trips)
    
    # Create IN clause with proper escaping
    trip_placeholders = ','.join('?' * len(active_trips_sample))
    
    query = f"""
    SELECT DISTINCT 
        t.route_id,
        t.trip_id,
        st.departure_time,
        r.route_short_name
    FROM stop_times st
    JOIN trips t ON st.trip_id = t.trip_id
    JOIN routes r ON t.route_id = r.route_id
    WHERE st.stop_id = ?
      AND st.departure_time >= ?
      AND st.departure_time <= ?
      AND st.trip_id IN ({trip_placeholders})
    ORDER BY st.departure_time
    LIMIT 50
    """
    
    query_params = [stop_id, earliest_time_str, latest_time_str] + active_trips_sample
    cursor.execute(query, query_params)
    
    results = []
    for route_id, trip_id, departure_time_str, route_short_name in cursor.fetchall():
        results.append({
            'route_id': route_id,
            'trip_id': trip_id, 
            'departure_time': time_str_to_seconds(departure_time_str),
            'route_short_name': route_short_name
        })
    
    return results


def query_line_segment_reachable(conn: sqlite3.Connection, trip_id: str, origin_stop: str,
                               departure_time: int, max_travel_time: int) -> List[Dict]:
    """Find all stops reachable on a specific trip within time limit"""
    cursor = conn.cursor()
    
    # Find the origin stop's sequence number
    cursor.execute("""
        SELECT stop_sequence FROM stop_times 
        WHERE trip_id = ? AND stop_id = ?
    """, (trip_id, origin_stop))
    
    origin_result = cursor.fetchone()
    if not origin_result:
        return []
    
    origin_sequence = origin_result[0]
    max_arrival_time = departure_time + max_travel_time
    max_arrival_str = seconds_to_time_str(max_arrival_time)
    
    # Find all subsequent stops on this trip within time limit
    cursor.execute("""
        SELECT stop_id, arrival_time, stop_sequence
        FROM stop_times
        WHERE trip_id = ?
          AND stop_sequence > ?
          AND arrival_time <= ?
        ORDER BY stop_sequence
    """, (trip_id, origin_sequence, max_arrival_str))
    
    results = []
    for stop_id, arrival_time_str, stop_sequence in cursor.fetchall():
        arrival_time = time_str_to_seconds(arrival_time_str)
        if arrival_time <= max_arrival_time:
            results.append({
                'stop_id': stop_id,
                'arrival_time': arrival_time,
                'stop_sequence': stop_sequence
            })
    
    return results


def query_transfers_on_line_segment(conn: sqlite3.Connection, trip_id: str, 
                                  origin_stop: str, max_travel_time: int) -> List[Dict]:
    """Find transfer opportunities along a trip segment"""
    cursor = conn.cursor()
    
    # Get origin sequence and departure time
    cursor.execute("""
        SELECT stop_sequence, departure_time FROM stop_times 
        WHERE trip_id = ? AND stop_id = ?
    """, (trip_id, origin_stop))
    
    origin_result = cursor.fetchone()
    if not origin_result:
        return []
    
    origin_sequence, origin_departure_str = origin_result
    origin_departure_time = time_str_to_seconds(origin_departure_str)
    max_arrival_time = origin_departure_time + max_travel_time
    max_arrival_str = seconds_to_time_str(max_arrival_time)
    
    # Find transfers available from stops on this trip segment
    cursor.execute("""
        SELECT 
            st.stop_id,
            st.arrival_time,
            tr.to_stop_id,
            tr.min_transfer_time
        FROM stop_times st
        JOIN transfers tr ON st.stop_id = tr.from_stop_id
        WHERE st.trip_id = ?
          AND st.stop_sequence > ?
          AND st.arrival_time <= ?
        ORDER BY st.stop_sequence
        LIMIT 20
    """, (trip_id, origin_sequence, max_arrival_str))
    
    results = []
    for stop_id, arrival_time_str, to_stop_id, min_transfer_time in cursor.fetchall():
        arrival_time = time_str_to_seconds(arrival_time_str)
        results.append({
            'stop_id': stop_id,
            'arrival_time': arrival_time,
            'to_stop_id': to_stop_id,
            'min_transfer_time': min_transfer_time
        })
    
    return results



def add_schedule_based_walking_expansion(reachable_stops: Dict[str, Dict], conn: sqlite3.Connection,
                                       stop_lines_mapping: Dict[str, set],
                                       max_walk_time_minutes: int = 20, max_total_time_minutes: int = 60) -> Tuple[Dict[str, Dict], Dict[str, Polygon]]:
    """
    Add walking expansion for schedule-based routing results with circle unions
    """
    expanded_points = {}
    cursor = conn.cursor()
    
    console.print(f"[cyan]Adding end-of-journey walking expansion for schedule-based results...[/cyan]")
    
    # Get stop coordinates from database
    stop_coords = {}
    cursor.execute("SELECT stop_id, stop_lat, stop_lon, stop_name FROM stops")
    for stop_id, lat, lon, name in cursor.fetchall():
        stop_coords[stop_id] = {'lat': lat, 'lon': lon, 'name': name}
    
    # Create walking circles for union computation
    from shapely.geometry import Point
    from shapely.ops import unary_union
    import pyproj
    
    # Set up coordinate transformation
    central_lat = sum(stop_coords[s]['lat'] for s in reachable_stops if s in stop_coords) / len(reachable_stops)
    central_lon = sum(stop_coords[s]['lon'] for s in reachable_stops if s in stop_coords) / len(reachable_stops)
    
    utm_zone = int((central_lon + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"
    
    wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    
    # Define time thresholds for cumulative zones
    time_thresholds = [10, 20, 30, 45, 60]
    time_range_names = ['very_close', 'close', 'medium', 'far', 'very_far']
    
    # Collect all circles with their total travel times for cumulative approach
    all_circles_with_times = []
    
    # Process each reachable stop
    for stop_id, stop_info in reachable_stops.items():
        if stop_id not in stop_coords:
            continue
            
        stop_data = stop_coords[stop_id]
        
        # Add the stop itself
        point_key = f"stop_{stop_id}"
        expanded_points[point_key] = {
            **stop_info,
            'point_type': 'transit_stop',
            'lat': stop_data['lat'],
            'lon': stop_data['lon'],
            'name': stop_data['name'],
            'stop_lat': stop_data['lat'],
            'stop_lon': stop_data['lon'],
            'stop_name': stop_data['name'],
            'lines': list(stop_lines_mapping.get(stop_id, set()))  # Add line information
        }
        
        # Calculate remaining time for walking and create circle
        transit_time = stop_info.get('total_time_minutes', stop_info.get('total_time', 0))
        remaining_time = max_total_time_minutes - transit_time
        
        if remaining_time > 0:
            # Calculate walking radius
            available_walk_time = min(remaining_time, max_walk_time_minutes)
            walk_radius_km = (available_walk_time / 60.0) * 5.0
            walk_radius_meters = walk_radius_km * 1000
            
            # Transform stop coordinates to UTM for accurate buffering
            utm_x, utm_y = wgs84_to_utm.transform(stop_data['lon'], stop_data['lat'])
            
            # Create circle in UTM coordinates
            stop_point_utm = Point(utm_x, utm_y)
            circle_utm = stop_point_utm.buffer(walk_radius_meters)
            
            # Transform circle back to WGS84
            circle_wgs84 = transform(utm_to_wgs84.transform, circle_utm)
            
            # Store circle with its total travel time (transit + walk to reach the circle edge)
            # Note: We use transit_time because the circle represents areas walkable FROM this transit stop
            all_circles_with_times.append((circle_wgs84, transit_time))
    
    # Create cumulative unions (areas reachable within X minutes total)
    def create_cumulative_union(max_time):
        """Create union of all circles for stops reachable within max_time minutes"""
        relevant_circles = [circle for circle, time in all_circles_with_times if time <= max_time]
        return safe_union(relevant_circles) if relevant_circles else None
    
    # Compute cumulative unions
    cumulative_unions = {}
    for i, threshold in enumerate(time_thresholds):
        if threshold <= max_total_time_minutes:
            union = create_cumulative_union(threshold)
            if union:
                cumulative_unions[threshold] = union
    
    # Create non-overlapping zones by subtracting inner areas from outer areas
    union_polygons = {}
    total_circles = len(all_circles_with_times)
    console.print(f"[dim]Computing non-overlapping unions for {total_circles} walking circles across {len(cumulative_unions)} time zones[/dim]")
    
    previous_union = None
    for i, threshold in enumerate(time_thresholds):
        if threshold not in cumulative_unions:
            continue
            
        range_name = time_range_names[i]
        current_union = cumulative_unions[threshold]
        
        if previous_union is None:
            # First zone - no subtraction needed
            zone_union = current_union
            zone_desc = f"0-{threshold}min"
        else:
            # Subsequent zones - subtract all previous areas
            zone_union = safe_difference(current_union, previous_union)
            prev_threshold = time_thresholds[i-1]
            zone_desc = f"{prev_threshold}-{threshold}min"
        
        if zone_union:
            union_polygons[range_name] = zone_union
            
            # Calculate area for reporting
            if hasattr(zone_union, 'area'):
                area_approx = zone_union.area * 111.0 * 111.0  # Rough conversion to km²
                cumulative_area = current_union.area * 111.0 * 111.0 if hasattr(current_union, 'area') else 0
                console.print(f"[green]✓ {range_name}: {zone_desc} → {area_approx:.1f} km² zone (cumulative: {cumulative_area:.1f} km²)[/green]")
        
        # Update previous union for next iteration
        previous_union = current_union
    
    console.print(f"[green]Created {len(union_polygons)} non-overlapping time-based walking area zones[/green]")
    console.print(f"[green]Expanded to {len(expanded_points)} transit stops with line information[/green]")
    
    return expanded_points, union_polygons


def parse_gtfs_time(time_str: str) -> int:
    """
    Parse GTFS time format (HH:MM:SS or H:MM:SS) to seconds since midnight
    Handles times > 24:00:00 for trips that span midnight
    Supports both zero-padded (08:00:00) and single-digit (8:00:00) formats
    """
    if not time_str or time_str.strip() == '':
        return 0
    hours, mins, secs = map(int, time_str.split(':'))
    return hours * 3600 + mins * 60 + secs


def time_str_to_seconds(time_str: str) -> int:
    """
    Convert time string (HH:MM:SS or H:MM:SS) to seconds since midnight
    Handles both zero-padded and single-digit hour formats
    """
    return parse_gtfs_time(time_str)


def seconds_to_time_str(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM:SS format"""
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def get_database_date_range(conn: sqlite3.Connection) -> Tuple[Optional[str], Optional[str]]:
    """Get the date range covered by the database"""
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT MIN(date), MAX(date) FROM calendar_dates WHERE exception_type = 1')
        result = cursor.fetchone()
        return result if result and result[0] and result[1] else (None, None)
    except:
        return (None, None)


def find_good_tuesday(conn: sqlite3.Connection) -> Optional[str]:
    """Find a Tuesday that has active services and isn't a holiday"""
    min_date, max_date = get_database_date_range(conn)
    if not min_date or not max_date:
        return None
    
    # Start from min_date and find the first Tuesday
    start_date = datetime.strptime(min_date, '%Y%m%d')
    
    # Find first Tuesday at or after start_date
    days_until_tuesday = (1 - start_date.weekday()) % 7  # Tuesday is weekday 1
    current_date = start_date + timedelta(days=days_until_tuesday)
    
    cursor = conn.cursor()
    end_date = datetime.strptime(max_date, '%Y%m%d')
    
    # Look for a Tuesday with active services (try up to 10 Tuesdays)
    for week in range(10):
        if current_date > end_date:
            break
            
        date_str = current_date.strftime('%Y%m%d')
        
        # Check if this date has active services
        cursor.execute('''
            SELECT COUNT(*) FROM calendar_dates 
            WHERE date = ? AND exception_type = 1
        ''', (date_str,))
        
        count = cursor.fetchone()[0]
        if count > 0:
            return date_str
            
        current_date += timedelta(days=7)  # Next Tuesday
    
    return None


def validate_date_coverage(conn: sqlite3.Connection, date_str: str) -> bool:
    """Check if the given date has active services in the database"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) FROM calendar_dates 
        WHERE date = ? AND exception_type = 1
    ''', (date_str,))
    
    count = cursor.fetchone()[0]
    return count > 0


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
    
    # Stops table - includes columns from both VRR and DELFI datasets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stops (
            stop_id TEXT PRIMARY KEY,
            stop_code TEXT,
            stop_name TEXT NOT NULL,
            stop_desc TEXT,
            stop_lat REAL NOT NULL,
            stop_lon REAL NOT NULL,
            stop_url TEXT,
            location_type INTEGER DEFAULT 0,
            parent_station TEXT,
            wheelchair_boarding INTEGER,
            platform_code TEXT,
            level_id TEXT,
            NVBW_HST_DHID TEXT
        )
    """)
    
    # Routes table - includes columns from both VRR and DELFI datasets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS routes (
            route_id TEXT PRIMARY KEY,
            agency_id TEXT,
            route_short_name TEXT,
            route_long_name TEXT,
            route_type INTEGER NOT NULL,
            route_color TEXT,
            route_text_color TEXT,
            route_desc TEXT,
            NVBW_DLID TEXT
        )
    """)
    
    # Trips table - includes columns from both VRR and DELFI datasets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            trip_id TEXT PRIMARY KEY,
            route_id TEXT NOT NULL,
            service_id TEXT NOT NULL,
            trip_headsign TEXT,
            trip_short_name TEXT,
            direction_id INTEGER,
            block_id TEXT,
            shape_id TEXT,
            wheelchair_accessible INTEGER,
            bikes_allowed INTEGER,
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
    
    # Transfers table - includes columns from both VRR and DELFI datasets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            from_stop_id TEXT NOT NULL,
            to_stop_id TEXT NOT NULL,
            transfer_type INTEGER NOT NULL,
            min_transfer_time INTEGER,
            from_route_id TEXT,
            to_route_id TEXT,
            from_trip_id TEXT,
            to_trip_id TEXT,
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
    
    # Calendar table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calendar (
            service_id TEXT PRIMARY KEY,
            monday INTEGER NOT NULL,
            tuesday INTEGER NOT NULL,
            wednesday INTEGER NOT NULL,
            thursday INTEGER NOT NULL,
            friday INTEGER NOT NULL,
            saturday INTEGER NOT NULL,
            sunday INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL
        )
    """)
    
    # Calendar dates table (exceptions)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calendar_dates (
            service_id TEXT NOT NULL,
            date TEXT NOT NULL,
            exception_type INTEGER NOT NULL,
            PRIMARY KEY (service_id, date),
            FOREIGN KEY (service_id) REFERENCES calendar(service_id)
        )
    """)
    
    # Create indexes for better query performance
    # Existing indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_lat_lon ON stops(stop_lat, stop_lon)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_stop_id ON stop_times(stop_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trips_route_id ON trips(route_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_calendar_dates_date ON calendar_dates(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trips_service_id ON trips(service_id)")
    
    # Performance-critical indexes for database-driven routing algorithm
    console.print("[cyan]Creating performance indexes for routing queries...[/cyan]")
    
    # Critical: Speed up trip departures lookup (most frequent query)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_stop_trip ON stop_times(stop_id, trip_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_trip_sequence ON stop_times(trip_id, stop_sequence)")
    
    # Critical: Speed up service filtering for active trips on date
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_calendar_dates_service_date ON calendar_dates(service_id, date)")
    
    # Important: Speed up route information lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trips_service_route ON trips(service_id, route_id)")
    
    # Important: Speed up transfer lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_from_stop ON transfers(from_stop_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_to_stop ON transfers(to_stop_id)")
    
    # Useful: Speed up line coverage optimization queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_stop_trip_route ON stop_times(stop_id, trip_id)")
    
    # Useful: Speed up geographic queries for finding walkable stops
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_lat ON stops(stop_lat)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_lon ON stops(stop_lon)")
    
    console.print("[green]Schema and performance indexes created successfully[/green]")


def import_gtfs_data(conn: sqlite3.Connection, data_dir: str) -> Dict[str, int]:
    """Import GTFS data from CSV/TXT files"""
    stats = {}
    data_path = Path(data_dir)
    
    # Define core GTFS file mappings - try both .csv and .txt extensions
    core_file_mappings = [
        ('stops', 'stops', ',', 'utf-8'),
        ('routes', 'routes', ',', 'utf-8'),
        ('trips', 'trips', ',', 'utf-8'),
        ('stop_times', 'stop_times', ',', 'utf-8'),
        ('transfers', 'transfers', ',', 'utf-8'),
        ('calendar', 'calendar', ',', 'utf-8'),
        ('calendar_dates', 'calendar_dates', ',', 'utf-8'),
    ]
    
    # VRR-specific files (optional)
    optional_file_mappings = [
        ('haltestellen', 'haltestellen', ';', 'latin-1'),
        ('linien', 'linien', ';', 'latin-1'),
    ]
    
    def find_file_with_extensions(base_name: str, extensions: List[str]) -> Optional[Path]:
        """Find file with any of the given extensions"""
        for ext in extensions:
            file_path = data_path / f"{base_name}.{ext}"
            if file_path.exists():
                return file_path
        return None
    
    # Import core GTFS files
    for base_name, table_name, delimiter, encoding in core_file_mappings:
        file_path = find_file_with_extensions(base_name, ['csv', 'txt'])
        if file_path:
            console.print(f"[cyan]Importing {file_path.name}...[/cyan]")
            count = import_csv_file(conn, file_path, table_name, delimiter, encoding)
            stats[table_name] = count
            console.print(f"[green]Imported {count:,} records from {file_path.name}[/green]")
        else:
            console.print(f"[red]Error: Required file {base_name} not found (tried .csv and .txt)[/red]")
    
    # Import optional files (skip silently if not found)
    for base_name, table_name, delimiter, encoding in optional_file_mappings:
        file_path = find_file_with_extensions(base_name, ['csv', 'txt'])
        if file_path:
            console.print(f"[cyan]Importing {file_path.name}...[/cyan]")
            count = import_csv_file(conn, file_path, table_name, delimiter, encoding)
            stats[table_name] = count
            console.print(f"[green]Imported {count:,} records from {file_path.name}[/green]")
    
    return stats


def import_csv_file(conn: sqlite3.Connection, file_path: Path, table_name: str, 
                   delimiter: str, encoding: str) -> int:
    """Import a single CSV file into the database with data quality fixes"""
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
        
        # Apply data quality fixes based on table type
        def apply_data_quality_fixes(row_dict: Dict) -> Optional[Dict]:
            """Apply data quality fixes to a row, return None to skip row"""
            
            # Geographic bounds filtering for stops table
            if table_name == 'stops':
                try:
                    lat = float(row_dict.get('stop_lat', 0))
                    lon = float(row_dict.get('stop_lon', 0))
                    
                    # Filter out stops outside reasonable German bounds
                    # Germany roughly: lat 47-55, lon 5-15
                    if not (45 <= lat <= 57 and 3 <= lon <= 17):
                        return None  # Skip this stop
                except (ValueError, TypeError):
                    return None  # Skip malformed coordinates
            
            # Route name normalization for routes table
            elif table_name == 'routes':
                # Ensure we have at least one name field
                short_name = row_dict.get('route_short_name', '').strip()
                long_name = row_dict.get('route_long_name', '').strip()
                route_id = row_dict.get('route_id', '').strip()
                
                # If both names are empty, use route_id as short_name
                if not short_name and not long_name:
                    row_dict['route_short_name'] = route_id
                elif not short_name and long_name:
                    # If only long_name exists, copy it to short_name for consistency
                    row_dict['route_short_name'] = long_name[:20]  # Truncate if too long
            
            return row_dict
        
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
                    # Apply data quality fixes to each row in chunk
                    cleaned_records = []
                    for _, row in chunk.iterrows():
                        row_dict = row.to_dict()
                        cleaned_row = apply_data_quality_fixes(row_dict)
                        if cleaned_row is not None:  # Keep row if not filtered out
                            cleaned_records.append([cleaned_row.get(col, None) for col in columns])
                    
                    if cleaned_records:
                        cursor.executemany(insert_sql, cleaned_records)
                        total_rows += len(cleaned_records)
                    
                    # Update progress based on file position
                    progress.update(task, advance=chunk_size * len(chunk.columns) * 50)
                    
            return total_rows
        else:
            # For smaller files, use regular csv reader with data quality fixes
            records = []
            skipped_count = 0
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    cleaned_row = apply_data_quality_fixes(row)
                    if cleaned_row is not None:  # Keep row if not filtered out
                        values = [cleaned_row.get(col, None) for col in columns]
                        records.append(values)
                    else:
                        skipped_count += 1
            
            if records:
                cursor.executemany(insert_sql, records)
            
            # Report data quality filtering
            if skipped_count > 0:
                console.print(f"[yellow]Data quality: Filtered out {skipped_count:,} records from {table_name}[/yellow]")
            
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
            
            # Use robust route mapping
            route_mappings, reverse_mappings = get_route_mappings(conn)
            
            # Try exact match first
            route_id = route_mappings.get(route)
            if route_id:
                console.print(f"[green]Found exact match: {route} -> {route_id}[/green]")
                routes_to_check = [(route_id, route)]
            else:
                # Fallback to pattern matching
                cursor.execute("""
                    SELECT route_id, route_short_name, route_long_name, route_type 
                    FROM routes 
                    WHERE route_short_name LIKE ? OR route_long_name LIKE ? OR route_id LIKE ?
                """, (f'%{route}%', f'%{route}%', f'%{route}%'))
                
                routes = cursor.fetchall()
                if routes:
                    console.print(f"[green]Found {len(routes)} matching routes via pattern search:[/green]")
                    routes_to_check = [(r[0], r[1] or r[0]) for r in routes[:3]]
                    for route_id, short_name, long_name, route_type in routes:
                        console.print(f"  • {short_name or 'N/A'} - {long_name or 'N/A'} (ID: {route_id}, Type: {route_type})")
                else:
                    console.print(f"[yellow]No routes found matching '{route}'[/yellow]")
                    console.print(f"[dim]Available routes: {', '.join(sorted(route_mappings.keys())[:10])}...[/dim]")
                    routes_to_check = []
            
            # For each route, find stops and check service dates
            for route_id, route_name in routes_to_check:
                console.print(f"\n[cyan]Stops for route {route_name} (ID: {route_id}):[/cyan]")
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
                    cursor.execute("""
                        SELECT COUNT(DISTINCT s.stop_id)
                        FROM stops s
                        JOIN stop_times st ON s.stop_id = st.stop_id
                        JOIN trips t ON st.trip_id = t.trip_id
                        WHERE t.route_id = ?
                    """, (route_id,))
                    total_stops = cursor.fetchone()[0]
                    console.print(f"    ... and {total_stops - 10} more stops (total: {total_stops})")
                
                # Check service information for this route
                cursor.execute("""
                    SELECT DISTINCT t.service_id
                    FROM trips t
                    WHERE t.route_id = ?
                """, (route_id,))
                
                service_ids = [row[0] for row in cursor.fetchall()]
                if service_ids:
                    console.print(f"    [dim]Service IDs: {', '.join(service_ids)}[/dim]")
                    
                    # Check if route runs on July 30, 2025
                    july30_active = any(is_service_active_on_date(conn, sid, '20250730') for sid in service_ids)
                    status = "✓ ACTIVE" if july30_active else "✗ INACTIVE"
                    console.print(f"    [dim]July 30, 2025: {status}[/dim]")
                else:
                    console.print(f"    [yellow]No service IDs found for this route[/yellow]")
        
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
@click.option('--date', help='Travel date (YYYYMMDD format, auto-selects good Tuesday if not specified)')
@click.option('--departure', default='08:00', help='Departure time (HH:MM format)')
@click.option('--visualize', is_flag=True, help='Generate interactive map after calculation')
@click.option('--map-output', default='isochrone_map.html', help='Output HTML file for map (if --visualize)')
@click.option('--verbose', is_flag=True, help='Print detailed line-following algorithm progress')
def query(db_path: str, address: str, lat: float, lon: float, max_time: int, date: str, departure: str, visualize: bool, map_output: str, verbose: bool):
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
            console.print(f"[dim]Coordinates: {start_lat:.6f}, {start_lon:.6f}[/dim]")
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
        # Find stops within walking distance
        console.print(f"\n[cyan]Finding stops within 20-minute walk of starting point...[/cyan]")
        walkable_stops = find_walkable_stops(conn, start_lat, start_lon, max_walk_time_minutes=20)
        
        if not walkable_stops:
            console.print("[yellow]No stops found within 20-minute walk of starting location[/yellow]")
            return
        
        console.print(f"[green]Found {len(walkable_stops)} stops within walking distance[/green]")
        console.print(f"[dim]Closest: {walkable_stops[0]['stop_name']} ({walkable_stops[0]['walk_time_minutes']:.1f}min walk)[/dim]")
        
        # Get stop-line mappings for the entire database
        console.print(f"\n[cyan]Optimizing origins using line coverage analysis...[/cyan]")
        stop_lines_mapping = get_stop_lines_mapping(conn)
        optimized_walkable_stops = optimize_walkable_stops_by_line_coverage(walkable_stops, stop_lines_mapping)
        
        # Convert departure time to seconds since midnight
        try:
            hours, minutes = map(int, departure.split(':'))
            start_time_seconds = hours * 3600 + minutes * 60
        except:
            console.print(f"[red]Invalid departure time format: {departure}. Use HH:MM format.[/red]")
            return
        
        # Handle date auto-selection and validation
        # Track if we auto-selected the date for better user feedback
        auto_selected_date = False
        if not date:
            # Auto-select a good Tuesday
            console.print("[cyan]No date specified, auto-selecting optimal date...[/cyan]")
            date = find_good_tuesday(conn)
            if not date:
                console.print("[red]Could not find a suitable date in the database[/red]")
                return
            auto_selected_date = True
        
        # Parse and validate date
        try:
            date_obj = datetime.strptime(date, '%Y%m%d')
        except:
            console.print(f"[red]Invalid date format: {date}. Use YYYYMMDD format.[/red]")
            return
        
        # Check if date has coverage in database
        if not validate_date_coverage(conn, date):
            min_date, max_date = get_database_date_range(conn)
            console.print(f"[red]⚠️  WARNING: No services found for {date} ({date_obj.strftime('%A, %B %d, %Y')})[/red]")
            if min_date and max_date:
                min_obj = datetime.strptime(min_date, '%Y%m%d')
                max_obj = datetime.strptime(max_date, '%Y%m%d')
                console.print(f"[yellow]Database covers: {min_obj.strftime('%B %d, %Y')} to {max_obj.strftime('%B %d, %Y')}[/yellow]")
                
                # Suggest a good alternative
                good_date = find_good_tuesday(conn)
                if good_date:
                    good_obj = datetime.strptime(good_date, '%Y%m%d')
                    console.print(f"[green]Suggestion: Use --date {good_date} ({good_obj.strftime('%A, %B %d, %Y')})[/green]")
            return
        
        # Display reference date and time prominently
        if auto_selected_date:
            console.print(f"\n[bold green]🗓️  Using auto-selected: {date_obj.strftime('%A, %B %d, %Y')} at {departure}[/bold green]")
            console.print(f"[dim]Auto-selected optimal Tuesday with active transit services[/dim]")
        else:
            console.print(f"\n[bold]🗓️  Reference: {date_obj.strftime('%A, %B %d, %Y')} at {departure}[/bold]")
            console.print(f"[dim]User-specified date and departure time[/dim]")
        
        # Use database-driven line-following algorithm
        console.print(f"\n[bold cyan]Using database-driven line-following algorithm for {date}[/bold cyan]")
        
        reachable_stops = database_line_following_isochrone(
            conn, optimized_walkable_stops, start_time_seconds, max_time, date, verbose=verbose
        )
        
        if not reachable_stops:
            console.print("[yellow]No stops reachable within time limit[/yellow]")
            return
        
        console.print(f"\n[bold green]Found {len(reachable_stops)} reachable stops within {max_time} minutes![/bold green]")
        
        # Add end-of-journey walking expansion using circle unions
        expanded_points, union_polygons = add_schedule_based_walking_expansion(
            reachable_stops, conn, stop_lines_mapping, max_walk_time_minutes=20, max_total_time_minutes=max_time
        )
        
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
                if min_time <= info.get('total_time_minutes', info.get('total_time', 0)) <= max_range
            ]
            
            if points_in_range:
                transit_stops = [p for p in points_in_range if p[1]['point_type'] == 'transit_stop']
                walking_points = [p for p in points_in_range if p[1]['point_type'] == 'walking_destination']
                
                console.print(f"\n[bold]{label}: {len(points_in_range)} total points[/bold]")
                console.print(f"[dim]  {len(transit_stops)} transit stops + {len(walking_points)} walking destinations[/dim]")
                
                # Show a few examples with breakdown
                examples = sorted(points_in_range, key=lambda x: x[1].get('total_time_minutes', x[1].get('total_time', 0)))[:3]
                for point_id, info in examples:
                    if info['point_type'] == 'transit_stop':
                        total_time_display = info.get('total_time_minutes', info.get('total_time', 0))
                        console.print(f"  🚏 {info.get('stop_name', point_id)}: {total_time_display:.1f}min (walk {info.get('walk_time', 0):.1f} + transit {info.get('transit_time', 0):.1f})")
                    else:
                        console.print(f"  🚶 Walking destination: {info.get('total_time_minutes', 0):.1f}min total")
                        
                console.print(f"  ... and {len(points_in_range) - len(examples)} more points")
        
        console.print(f"\n[dim]Summary:[/dim]")
        console.print(f"  Transit stops reachable: {len(reachable_stops)}")
        console.print(f"  Additional walking destinations: {len(expanded_points) - len(reachable_stops)}")
        console.print(f"  Total reachable area: {len(expanded_points)} points")
        
        # Generate visualization if requested
        if visualize:
            console.print(f"\n[cyan]Generating interactive map visualization...[/cyan]")
            
            title = f"Reachable Area within {max_time} minutes"
            if address:
                title = f"{title} from {address}"
            
            # Use circle union visualization with schedule-based data
            map_path = create_circle_union_map(
                start_lat, start_lon, union_polygons, reachable_stops, conn, stop_lines_mapping, 
                title, map_output, date, departure
            )
            
            if map_path:
                console.print(f"\n[bold green]✓ Interactive map generated![/bold green]")
                console.print(f"[green]Open in browser: file://{map_path}[/green]")
    
    finally:
        conn.close()
    
    console.print("\n[dim]Data © OpenStreetMap contributors[/dim]")


@cli.command()
@click.option('--vrr-db', default='data/vrr.db', help='Path to VRR database')
@click.option('--delfi-db', default='data/delfi.db', help='Path to DELFI database')
@click.option('--area', default='Dortmund', help='Focus area for comparison (city name)')
@click.option('--radius', default=50, help='Radius in km around focus area')
def compare_datasets(vrr_db: str, delfi_db: str, area: str, radius: int):
    """Compare VRR and DELFI datasets to identify missing connections"""
    
    console.print(f"[bold blue]Comparing datasets for {area} area (±{radius}km)[/bold blue]")
    
    # Check if both databases exist
    if not os.path.exists(vrr_db):
        console.print(f"[red]VRR database not found: {vrr_db}[/red]")
        return
    if not os.path.exists(delfi_db):
        console.print(f"[red]DELFI database not found: {delfi_db}[/red]")
        return
    
    # Geocode the focus area
    console.print(f"[cyan]Geocoding focus area: {area}[/cyan]")
    coords = geocode_address(area)
    if not coords:
        console.print(f"[red]Could not geocode area: {area}[/red]")
        return
    
    focus_lat, focus_lon = coords
    console.print(f"[dim]Focus coordinates: {focus_lat:.4f}, {focus_lon:.4f}[/dim]")
    
    # Connect to both databases
    vrr_conn = sqlite3.connect(vrr_db)
    delfi_conn = sqlite3.connect(delfi_db)
    
    try:
        # Get geographic bounds for filtering
        lat_min = focus_lat - (radius / 111.0)  # Rough conversion: 1 degree ≈ 111km
        lat_max = focus_lat + (radius / 111.0)
        lon_min = focus_lon - (radius / (111.0 * abs(math.cos(math.radians(focus_lat)))))
        lon_max = focus_lon + (radius / (111.0 * abs(math.cos(math.radians(focus_lat)))))
        
        console.print(f"[dim]Search bounds: {lat_min:.4f} to {lat_max:.4f}, {lon_min:.4f} to {lon_max:.4f}[/dim]")
        
        # Compare stop coverage
        console.print(f"\n[cyan]Analyzing stop coverage...[/cyan]")
        vrr_stops = get_stops_in_area(vrr_conn, lat_min, lat_max, lon_min, lon_max)
        delfi_stops = get_stops_in_area(delfi_conn, lat_min, lat_max, lon_min, lon_max)
        
        console.print(f"[green]VRR stops in area: {len(vrr_stops)}[/green]")
        console.print(f"[green]DELFI stops in area: {len(delfi_stops)}[/green]")
        
        # Find stops that exist in VRR but not in DELFI (by proximity)
        missing_in_delfi = find_missing_stops(vrr_stops, delfi_stops, threshold_meters=100)
        console.print(f"[yellow]Stops in VRR but missing/distant in DELFI: {len(missing_in_delfi)}[/yellow]")
        
        if missing_in_delfi:
            console.print(f"\n[bold]Sample missing stops:[/bold]")
            for i, stop in enumerate(missing_in_delfi[:10]):
                console.print(f"  {i+1}. {stop['stop_name']} ({stop['stop_lat']:.4f}, {stop['stop_lon']:.4f})")
            if len(missing_in_delfi) > 10:
                console.print(f"  ... and {len(missing_in_delfi) - 10} more")
        
        # Compare route coverage
        console.print(f"\n[cyan]Analyzing route coverage...[/cyan]")
        vrr_routes = get_routes_in_area(vrr_conn, lat_min, lat_max, lon_min, lon_max)
        delfi_routes = get_routes_in_area(delfi_conn, lat_min, lat_max, lon_min, lon_max)
        
        console.print(f"[green]VRR routes in area: {len(vrr_routes)}[/green]")
        console.print(f"[green]DELFI routes in area: {len(delfi_routes)}[/green]")
        
        # Find routes that exist in VRR but not in DELFI
        missing_routes = find_missing_routes(vrr_routes, delfi_routes)
        console.print(f"[yellow]Routes in VRR but missing in DELFI: {len(missing_routes)}[/yellow]")
        
        if missing_routes:
            console.print(f"\n[bold]Sample missing routes:[/bold]")
            for i, route in enumerate(missing_routes[:10]):
                route_name = route['route_short_name'] or route['route_long_name'] or route['route_id']
                console.print(f"  {i+1}. {route_name} (Type: {route['route_type']})")
            if len(missing_routes) > 10:
                console.print(f"  ... and {len(missing_routes) - 10} more")
        
        # Connectivity analysis
        console.print(f"\n[cyan]Testing connectivity between major stops...[/cyan]")
        analyze_connectivity_differences(vrr_conn, delfi_conn, focus_lat, focus_lon, radius=20)
        
        # Summary
        console.print(f"\n[bold blue]Summary for {area} area:[/bold blue]")
        console.print(f"  Coverage gap: {len(missing_in_delfi)} stops, {len(missing_routes)} routes")
        
        coverage_percentage = (1 - len(missing_in_delfi) / max(len(vrr_stops), 1)) * 100
        console.print(f"  DELFI stop coverage: {coverage_percentage:.1f}% of VRR stops")
        
        route_coverage_percentage = (1 - len(missing_routes) / max(len(vrr_routes), 1)) * 100
        console.print(f"  DELFI route coverage: {route_coverage_percentage:.1f}% of VRR routes")
        
        if coverage_percentage < 80:
            console.print(f"[yellow]⚠️  Significant coverage gap detected in DELFI dataset[/yellow]")
        else:
            console.print(f"[green]✓ Good coverage overlap between datasets[/green]")
            
    finally:
        vrr_conn.close()
        delfi_conn.close()


def get_stops_in_area(conn: sqlite3.Connection, lat_min: float, lat_max: float, 
                     lon_min: float, lon_max: float) -> List[Dict]:
    """Get all stops within geographic bounds"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT stop_id, stop_name, stop_lat, stop_lon
        FROM stops
        WHERE stop_lat BETWEEN ? AND ? 
        AND stop_lon BETWEEN ? AND ?
    """, (lat_min, lat_max, lon_min, lon_max))
    
    return [{'stop_id': row[0], 'stop_name': row[1], 
             'stop_lat': row[2], 'stop_lon': row[3]} for row in cursor.fetchall()]


def get_routes_in_area(conn: sqlite3.Connection, lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float) -> List[Dict]:
    """Get all routes that serve stops within geographic bounds"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT r.route_id, r.route_short_name, r.route_long_name, r.route_type
        FROM routes r
        JOIN trips t ON r.route_id = t.route_id
        JOIN stop_times st ON t.trip_id = st.trip_id
        JOIN stops s ON st.stop_id = s.stop_id
        WHERE s.stop_lat BETWEEN ? AND ?
        AND s.stop_lon BETWEEN ? AND ?
    """, (lat_min, lat_max, lon_min, lon_max))
    
    return [{'route_id': row[0], 'route_short_name': row[1],
             'route_long_name': row[2], 'route_type': row[3]} for row in cursor.fetchall()]


def find_missing_stops(vrr_stops: List[Dict], delfi_stops: List[Dict], 
                      threshold_meters: float = 100) -> List[Dict]:
    """Find VRR stops that don't have a close equivalent in DELFI"""
    missing = []
    
    for vrr_stop in vrr_stops:
        # Check if there's a DELFI stop within threshold distance
        closest_distance = float('inf')
        
        for delfi_stop in delfi_stops:
            distance = haversine_distance(
                vrr_stop['stop_lat'], vrr_stop['stop_lon'],
                delfi_stop['stop_lat'], delfi_stop['stop_lon']
            ) * 1000  # Convert to meters
            
            if distance < closest_distance:
                closest_distance = distance
        
        # If no DELFI stop is within threshold, consider it missing
        if closest_distance > threshold_meters:
            missing.append(vrr_stop)
    
    return missing


def find_missing_routes(vrr_routes: List[Dict], delfi_routes: List[Dict]) -> List[Dict]:
    """Find VRR routes that don't exist in DELFI by name/number"""
    missing = []
    
    # Create set of DELFI route identifiers for fast lookup
    delfi_identifiers = set()
    for route in delfi_routes:
        if route['route_short_name']:
            delfi_identifiers.add(route['route_short_name'].strip())
        if route['route_long_name']:
            delfi_identifiers.add(route['route_long_name'].strip())
    
    for vrr_route in vrr_routes:
        found = False
        
        # Check if VRR route exists in DELFI by short name or long name
        if vrr_route['route_short_name']:
            if vrr_route['route_short_name'].strip() in delfi_identifiers:
                found = True
        
        if not found and vrr_route['route_long_name']:
            if vrr_route['route_long_name'].strip() in delfi_identifiers:
                found = True
        
        if not found:
            missing.append(vrr_route)
    
    return missing


def analyze_connectivity_differences(vrr_conn: sqlite3.Connection, delfi_conn: sqlite3.Connection,
                                   focus_lat: float, focus_lon: float, radius: int = 20):
    """Analyze connectivity differences between datasets for major stops"""
    
    # Find major stops in focus area (high connection count)
    cursor = vrr_conn.cursor()
    cursor.execute("""
        SELECT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon, 
               COUNT(DISTINCT st.trip_id) as connection_count
        FROM stops s
        JOIN stop_times st ON s.stop_id = st.stop_id
        WHERE s.stop_lat BETWEEN ? AND ?
        AND s.stop_lon BETWEEN ? AND ?
        GROUP BY s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
        HAVING connection_count > 50
        ORDER BY connection_count DESC
        LIMIT 10
    """, (
        focus_lat - radius/111.0, focus_lat + radius/111.0,
        focus_lon - radius/111.0, focus_lon + radius/111.0
    ))
    
    major_stops = cursor.fetchall()
    
    if not major_stops:
        console.print("[yellow]No major stops found in focus area[/yellow]")
        return
    
    console.print(f"[green]Found {len(major_stops)} major stops for connectivity testing[/green]")
    
    # Test connectivity between pairs of major stops
    test_pairs = []
    for i in range(min(3, len(major_stops))):
        for j in range(i+1, min(3, len(major_stops))):
            test_pairs.append((major_stops[i], major_stops[j]))
    
    console.print(f"[dim]Testing {len(test_pairs)} stop pairs...[/dim]")
    
    for stop1, stop2 in test_pairs:
        stop1_name = stop1[1]
        stop2_name = stop2[1]
        
        # For now, just report the stops - full routing comparison would be complex
        console.print(f"  {stop1_name} ↔ {stop2_name}")
        console.print(f"    VRR connections: {stop1[4]}, {stop2[4]}")
        
        # Could add: actual route finding and comparison here
        # This would require implementing routing in both databases


if __name__ == '__main__':
    cli()
