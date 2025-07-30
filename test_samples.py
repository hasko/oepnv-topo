#!/usr/bin/env python3
"""Test sample routes against our algorithm"""

import subprocess
import re
import sys
from typing import Tuple, Optional

def parse_sample_line(line: str) -> Tuple[str, str, int]:
    """Parse a sample line into origin, destination, expected time"""
    parts = line.strip().split(';')
    if len(parts) != 3:
        return None
    origin = parts[0].strip()
    destination = parts[1].strip()
    expected_time = int(parts[2].strip())
    return origin, destination, expected_time

def geocode_and_find_time(origin: str, destination: str, algorithm: str = "") -> Optional[int]:
    """Run the algorithm and find travel time to destination"""
    # First geocode the destination to get its coordinates
    geocode_cmd = [
        "uv", "run", "python", "main.py", "query",
        "--address", destination,
        "--time", "1"  # Just to get coordinates
    ]
    
    result = subprocess.run(geocode_cmd, capture_output=True, text=True)
    
    # Extract destination name from output
    dest_pattern = r"Geocoding address: (.+)"
    dest_match = re.search(dest_pattern, result.stdout)
    if not dest_match:
        print(f"Could not geocode destination: {destination}")
        return None
    
    # Run the actual query with a reasonable time limit
    cmd = [
        "uv", "run", "python", "main.py", "query",
        "--address", origin,
        "--time", "60"  # 60 minutes should cover all test cases
    ]
    
    if algorithm:
        cmd.append(algorithm)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Look for the destination in the output
    # The output format includes lines like:
    # 🚏 Destination Name: XX.Xmin (walk X.X + transit X.X)
    
    # Try to find the destination stop in the output
    lines = result.stdout.split('\n')
    for line in lines:
        if '🚏' in line and any(part in line for part in destination.split(',')):
            # Extract time from format: "XX.Xmin"
            time_match = re.search(r': (\d+\.\d+)min', line)
            if time_match:
                return int(float(time_match.group(1)))
    
    # If exact match not found, try to find closest match
    # Look for the destination street name
    street_name = destination.split(',')[0].split()[0]  # Get first word of street
    for line in lines:
        if '🚏' in line and street_name in line:
            time_match = re.search(r': (\d+\.\d+)min', line)
            if time_match:
                travel_time = int(float(time_match.group(1)))
                print(f"  Found partial match: {line.strip()}")
                return travel_time
    
    return None

def main():
    """Run tests on all samples"""
    print("Testing sample routes with both algorithms...")
    print("=" * 80)
    
    with open('samples.txt', 'r') as f:
        samples = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        parsed = parse_sample_line(sample)
        if not parsed:
            continue
            
        origin, destination, expected_time = parsed
        
        print(f"\nTest {i}: {origin} → {destination}")
        print(f"Expected time: {expected_time} minutes")
        
        # Test with graph-based algorithm
        print("  Graph-based algorithm: ", end='', flush=True)
        graph_time = geocode_and_find_time(origin, destination, "")
        if graph_time:
            diff = graph_time - expected_time
            status = "✓" if abs(diff) <= 5 else "✗"
            print(f"{graph_time} min (diff: {diff:+d}) {status}")
        else:
            print("Destination not found")
        
        # Test with database-driven algorithm
        print("  Database-driven algorithm: ", end='', flush=True)
        db_time = geocode_and_find_time(origin, destination, "--database-driven")
        if db_time:
            diff = db_time - expected_time
            status = "✓" if abs(diff) <= 5 else "✗"
            print(f"{db_time} min (diff: {diff:+d}) {status}")
        else:
            print("Destination not found")
        
        results.append({
            'origin': origin,
            'destination': destination,
            'expected': expected_time,
            'graph': graph_time,
            'database': db_time
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"{'Route':<50} {'Expected':>10} {'Graph':>10} {'Database':>10}")
    print("-" * 80)
    
    for r in results:
        route = f"{r['origin'].split(',')[0]} → {r['destination'].split(',')[0]}"
        graph_str = f"{r['graph']} min" if r['graph'] else "N/A"
        db_str = f"{r['database']} min" if r['database'] else "N/A"
        print(f"{route:<50} {r['expected']:>10} {graph_str:>10} {db_str:>10}")

if __name__ == "__main__":
    main()