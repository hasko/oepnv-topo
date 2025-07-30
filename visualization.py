#!/usr/bin/env python3
"""
Visualization module for ÖPNV Topo - Creates interactive maps with isochrone overlays
"""
import folium
import geopandas as gpd
import alphashape
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from rich.console import Console
import os

console = Console()


def points_to_polygon(points: List[Tuple[float, float]], alpha: float = 0.0) -> Optional[Polygon]:
    """
    Convert a list of (lat, lon) points to a polygon using alpha shapes.
    
    Args:
        points: List of (latitude, longitude) tuples
        alpha: Alpha parameter for shape (0.0 = convex hull, higher = more detailed)
    
    Returns:
        Shapely Polygon or None if conversion fails
    """
    if len(points) < 3:
        console.print("[yellow]Warning: Need at least 3 points to create polygon[/yellow]")
        return None
    
    try:
        # Ensure we have valid coordinate tuples
        valid_points = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                lat, lon = float(point[0]), float(point[1])
                # Note: shapely uses (x, y) = (lon, lat)
                valid_points.append((lon, lat))
            else:
                console.print(f"[dim]Skipping invalid point: {point}[/dim]")
        
        if len(valid_points) < 3:
            console.print(f"[yellow]Warning: Only {len(valid_points)} valid points after filtering[/yellow]")
            return None
        
        console.print(f"[dim]Creating polygon from {len(valid_points)} points[/dim]")
        
        # Generate alpha shape (alphashape expects coordinate tuples, not Point objects)
        alpha_shape = alphashape.alphashape(valid_points, alpha)
        
        # Handle MultiPolygon by taking the largest polygon
        if isinstance(alpha_shape, MultiPolygon):
            # Get the polygon with the largest area
            alpha_shape = max(alpha_shape.geoms, key=lambda p: p.area)
        
        return alpha_shape if isinstance(alpha_shape, Polygon) else None
        
    except Exception as e:
        console.print(f"[red]Error creating polygon: {e}[/red]")
        return None


def create_isochrone_map(
    center_lat: float,
    center_lon: float,
    reachable_points: Dict[str, Dict],
    title: str = "Public Transport Isochrone",
    output_file: str = "isochrone_map.html"
) -> str:
    """
    Create an interactive map showing reachable areas as polygon overlays.
    
    Args:
        center_lat: Center latitude (starting point)
        center_lon: Center longitude (starting point)
        reachable_points: Dictionary of reachable stops with coordinates and times
        title: Map title
        output_file: Output HTML filename
    
    Returns:
        Path to generated HTML file
    """
    console.print(f"[cyan]Creating interactive map: {title}[/cyan]")
    
    # Initialize map centered on starting location
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = f'''
                 <h3 align="center" style="font-size:16px; margin-top:10px;"><b>{title}</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add center marker
    folium.Marker(
        [center_lat, center_lon],
        popup="Starting Point",
        tooltip="Starting Point",
        icon=folium.Icon(color='red', icon='play')
    ).add_to(m)
    
    # Group points by time ranges for different colored overlays
    # Using magenta/purple colors to avoid conflict with green areas on OSM
    time_ranges = [
        (0, 10, '#ff00ff', 'magenta'),      # 0-10min: bright magenta
        (10, 20, '#dd44dd', 'mediumorchid'), # 10-20min: medium orchid  
        (20, 30, '#bb88bb', 'plum'),        # 20-30min: plum
        (30, 45, '#cc99cc', 'thistle'),     # 30-45min: thistle
        (45, 60, '#e6d6e6', 'lavender')     # 45-60min: light lavender
    ]
    
    # Process each time range
    for min_time, max_time, color, color_name in time_ranges:
        # Filter points for this time range
        range_points = []
        for stop_id, data in reachable_points.items():
            travel_time = data.get('travel_time', float('inf'))
            if min_time <= travel_time < max_time:
                lat = data.get('stop_lat')
                lon = data.get('stop_lon')
                if lat is not None and lon is not None:
                    range_points.append((lat, lon))
        
        if len(range_points) < 3:
            console.print(f"[dim]Skipping {min_time}-{max_time}min range: only {len(range_points)} points[/dim]")
            continue
        
        # Create polygon for this time range
        polygon = points_to_polygon(range_points, alpha=0.1)
        
        if polygon and polygon.is_valid:
            # Convert to GeoDataFrame for Folium
            gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
            
            # Add polygon to map
            folium.GeoJson(
                gdf.geometry.iloc[0],
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.3,
                    'opacity': 0.6
                },
                popup=f"Reachable in {min_time}-{max_time} minutes",
                tooltip=f"{min_time}-{max_time} min"
            ).add_to(m)
            
            console.print(f"[magenta]Added {color_name} zone: {len(range_points)} points ({min_time}-{max_time}min)[/magenta]")
    
    # Add individual transit stops as small markers
    transit_stops = 0
    for stop_id, data in reachable_points.items():
        if data.get('type') == 'transit':
            lat = data.get('stop_lat')
            lon = data.get('stop_lon') 
            travel_time = data.get('travel_time', 0)
            stop_name = data.get('stop_name', f'Stop {stop_id}')
            
            if lat is not None and lon is not None:
                folium.CircleMarker(
                    [lat, lon],
                    radius=3,
                    popup=f"{stop_name}<br>Travel time: {travel_time:.1f} min",
                    tooltip=f"{stop_name} ({travel_time:.1f}min)",
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.7
                ).add_to(m)
                transit_stops += 1
    
    console.print(f"[blue]Added {transit_stops} transit stop markers[/blue]")
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px
                ">
    <p><b>Travel Time Zones</b></p>
    <p><i class="fa fa-circle" style="color:#ff00ff"></i> 0-10 minutes</p>
    <p><i class="fa fa-circle" style="color:#dd44dd"></i> 10-20 minutes</p>
    <p><i class="fa fa-circle" style="color:#bb88bb"></i> 20-30 minutes</p>
    <p><i class="fa fa-circle" style="color:#cc99cc"></i> 30-45 minutes</p>
    <p><i class="fa fa-circle" style="color:#e6d6e6"></i> 45-60 minutes</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    absolute_path = os.path.abspath(output_file)
    
    console.print(f"[green]✓ Map saved to: {absolute_path}[/green]")
    console.print(f"[dim]Open in browser: file://{absolute_path}[/dim]")
    
    return absolute_path



def create_simple_boundary_map(
    center_lat: float,
    center_lon: float,
    reachable_points: Dict[str, Dict],
    max_time: int,
    title: str = "Reachable Area",
    output_file: str = "boundary_map.html"
) -> str:
    """
    Create a simple map with single boundary polygon for all reachable points.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        reachable_points: Dictionary of reachable stops
        max_time: Maximum travel time for the boundary
        title: Map title
        output_file: Output HTML filename
    
    Returns:
        Path to generated HTML file
    """
    console.print(f"[cyan]Creating boundary map for {max_time}-minute reachable area[/cyan]")
    
    # Extract all reachable coordinates
    points = []
    for stop_id, data in reachable_points.items():
        lat = data.get('stop_lat')
        lon = data.get('stop_lon')
        if lat is not None and lon is not None:
            points.append((lat, lon))
    
    if len(points) < 3:
        console.print(f"[red]Error: Need at least 3 points, got {len(points)}[/red]")
        return ""
    
    # Create polygon
    polygon = points_to_polygon(points, alpha=0.05)
    
    if not polygon or not polygon.is_valid:
        console.print("[red]Error: Could not create valid polygon[/red]")
        return ""
    
    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = f'''
                 <h3 align="center" style="font-size:16px; margin-top:10px;"><b>{title}</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add starting point
    folium.Marker(
        [center_lat, center_lon],
        popup="Starting Point",
        tooltip="Starting Point",
        icon=folium.Icon(color='red', icon='play')
    ).add_to(m)
    
    # Add boundary polygon
    gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
    
    folium.GeoJson(
        gdf.geometry.iloc[0],
        style_function=lambda feature: {
            'fillColor': '#ff00ff',
            'color': '#cc00cc',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.8
        },
        popup=f"Reachable area within {max_time} minutes",
        tooltip=f"Reachable in {max_time} minutes"
    ).add_to(m)
    
    # Save map
    m.save(output_file)
    absolute_path = os.path.abspath(output_file)
    
    console.print(f"[green]✓ Boundary map saved to: {absolute_path}[/green]")
    console.print(f"[green]✓ Polygon covers {len(points)} reachable points[/green]")
    console.print(f"[dim]Open in browser: file://{absolute_path}[/dim]")
    
    return absolute_path


def create_circle_union_map(
    center_lat: float,
    center_lon: float,
    union_polygons: Dict[str, Polygon],
    reachable_stops: Dict[str, Dict],
    conn,
    stop_lines_mapping: Dict[str, set] = None,
    title: str = "Circle Union Isochrone",
    output_file: str = "circle_union_map.html",
    reference_date: str = None,
    departure_time: str = None
) -> str:
    """
    Create an interactive map showing union polygons directly instead of alpha shapes.
    This preserves the accurate circular boundaries from the union calculation.
    
    Args:
        center_lat: Center latitude (starting point)
        center_lon: Center longitude (starting point)
        union_polygons: Dictionary mapping time ranges to union polygon geometries
        reachable_stops: Dictionary of reachable transit stops for markers
        graph: NetworkX graph with stop coordinates
        stop_lines_mapping: Dictionary mapping stop_id to set of route_ids (optional)
        title: Map title
        output_file: Output HTML filename
        reference_date: Reference date in YYYYMMDD format
        departure_time: Departure time in HH:MM format
    
    Returns:
        Path to generated HTML file
    """
    console.print(f"[cyan]Creating circle union map: {title}[/cyan]")
    
    # Initialize map centered on starting location
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add title with optional date/time information
    title_html = f'''
                 <h3 align="center" style="font-size:16px; margin-top:10px;"><b>{title}</b></h3>
                 '''
    
    # Add reference date and time if provided
    if reference_date and departure_time:
        from datetime import datetime
        try:
            date_obj = datetime.strptime(reference_date, '%Y%m%d')
            subtitle_html = f'''
                         <p align="center" style="font-size:12px; margin-top:-5px; color:#666;">
                         Reference: {date_obj.strftime('%A, %B %d, %Y')} at {departure_time}
                         </p>
                         '''
            title_html += subtitle_html
        except:
            # If date parsing fails, just show the raw date
            subtitle_html = f'''
                         <p align="center" style="font-size:12px; margin-top:-5px; color:#666;">
                         Reference: {reference_date} at {departure_time}
                         </p>
                         '''
            title_html += subtitle_html
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add center marker
    folium.Marker(
        [center_lat, center_lon],
        popup="Starting Point",
        tooltip="Starting Point",
        icon=folium.Icon(color='red', icon='play')
    ).add_to(m)
    
    # Color mapping for time ranges
    range_colors = {
        'very_close': ('#ff00ff', 'bright magenta', '0-10 minutes'),      
        'close': ('#dd44dd', 'medium orchid', '10-20 minutes'),           
        'medium': ('#bb88bb', 'plum', '20-30 minutes'),                   
        'far': ('#cc99cc', 'thistle', '30-45 minutes'),                   
        'very_far': ('#e6d6e6', 'light lavender', '45-60 minutes')        
    }
    
    # Add union polygons directly to the map
    legend_items = []
    for range_name, polygon in union_polygons.items():
        if not polygon or polygon.is_empty:
            continue
            
        color, color_name, time_desc = range_colors.get(range_name, ('#888888', 'gray', 'unknown'))
        
        # Handle MultiPolygon by processing each polygon separately
        polygons_to_add = []
        if isinstance(polygon, MultiPolygon):
            polygons_to_add = list(polygon.geoms)
        else:
            polygons_to_add = [polygon]
        
        for poly in polygons_to_add:
            if poly.is_valid and not poly.is_empty:
                # Convert to GeoDataFrame for Folium
                gdf = gpd.GeoDataFrame([1], geometry=[poly], crs="EPSG:4326")
                
                # Add polygon to map with preserved circular boundaries
                folium.GeoJson(
                    gdf.geometry.iloc[0],
                    style_function=lambda feature, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 2,
                        'fillOpacity': 0.3,
                        'opacity': 0.6
                    },
                    popup=f"Walking area reachable in {time_desc}",
                    tooltip=f"{time_desc} (circle union)"
                ).add_to(m)
        
        # Calculate area for reporting
        area_approx = polygon.area * 111.0 * 111.0  # Rough conversion to km²
        console.print(f"[magenta]Added {color_name} union: {area_approx:.1f} km² ({time_desc})[/magenta]")
        legend_items.append((color, time_desc))
    
    # Add individual transit stops as markers
    transit_stops = 0
    cursor = conn.cursor()
    
    for stop_id, stop_info in reachable_stops.items():
        # Get coordinates from database
        cursor.execute("SELECT stop_lat, stop_lon, stop_name FROM stops WHERE stop_id = ?", (stop_id,))
        stop_data = cursor.fetchone()
        
        if stop_data:
            lat, lon, stop_name = stop_data
            travel_time = stop_info.get('total_time_minutes', 0)
            
            # Get line information if available
            lines_info = ""
            if stop_lines_mapping and stop_id in stop_lines_mapping:
                lines = sorted(list(stop_lines_mapping[stop_id]))
                if lines:
                    # Show first 4 lines, then indicate if there are more
                    if len(lines) <= 4:
                        lines_info = f"<br>Lines: {', '.join(lines)}"
                    else:
                        lines_info = f"<br>Lines: {', '.join(lines[:4])} (+{len(lines)-4} more)"
            
            if lat is not None and lon is not None:
                # Create tooltip text with line information
                tooltip_text = f"{stop_name} ({travel_time:.1f}min)"
                if lines_info:
                    # Extract just the line names for tooltip (no HTML)
                    lines_for_tooltip = sorted(list(stop_lines_mapping[stop_id])) if stop_lines_mapping and stop_id in stop_lines_mapping else []
                    if lines_for_tooltip:
                        if len(lines_for_tooltip) <= 3:
                            tooltip_text += f" • {', '.join(lines_for_tooltip)}"
                        else:
                            tooltip_text += f" • {', '.join(lines_for_tooltip[:3])}..."
                
                folium.CircleMarker(
                    [lat, lon],
                    radius=4,
                    popup=f"{stop_name}<br>Travel time: {travel_time:.1f} min{lines_info}",
                    tooltip=tooltip_text,
                    color='darkblue',
                    fillColor='lightblue',
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
                transit_stops += 1
    
    console.print(f"[blue]Added {transit_stops} transit stop markers[/blue]")
    
    # Add dynamic legend based on what's actually shown
    if legend_items:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 220px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px
                    ">
        <p><b>Walking Areas (Circle Unions)</b></p>
        '''
        for color, time_desc in legend_items:
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {time_desc}</p>'
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    absolute_path = os.path.abspath(output_file)
    
    console.print(f"[green]✓ Circle union map saved to: {absolute_path}[/green]")
    console.print(f"[green]✓ Shows {len(union_polygons)} union polygons with preserved boundaries[/green]")
    console.print(f"[dim]Open in browser: file://{absolute_path}[/dim]")
    
    return absolute_path