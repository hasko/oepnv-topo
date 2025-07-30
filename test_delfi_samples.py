#!/usr/bin/env python3
"""Test sample routes against DELFI database"""

import subprocess
import re
import sys
from typing import Tuple, Optional, List, Dict

def parse_sample_line(line: str) -> Tuple[str, str, int]:
    """Parse a sample line into origin, destination, expected time"""
    parts = line.strip().split(';')
    if len(parts) != 3:
        return None
    origin = parts[0].strip()
    destination = parts[1].strip()
    expected_time = int(parts[2].strip())
    return origin, destination, expected_time

def run_delfi_query(origin: str, destination: str, max_time: int = 60) -> Optional[Dict]:
    """Run query against DELFI database and find travel time to destination"""
    
    # Run the query command with DELFI database
    cmd = [
        "uv", "run", "python", "main.py", "query",
        "--db-path", "data/delfi.db",
        "--address", origin,
        "--time", str(max_time),
        "--date", "20250728"  # Valid date in DELFI coverage period (July 2025)
    ]
    
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    Error running command: {result.stderr}")
        return None
    
    # Parse the output to find destination and travel time
    lines = result.stdout.split('\n')
    
    # Look for the destination in reachable stops
    destination_parts = destination.replace(',', '').split()
    dest_lower = destination.lower()
    
    # Skip the origin stop by looking for Emil-Figge-Straße and filtering it out
    origin_lower = origin.lower()
    
    for line in lines:
        # Look for transit stop entries: "🚏 Stop Name: XX.Xmin (walk X.X + transit X.X)"
        if '🚏' in line and ':' in line and 'min' in line:
            line_lower = line.lower()
            
            # Skip origin stops
            if 'emil-figge' in line_lower or 'emil figge' in line_lower:
                continue
            
            # Try exact matching first
            if any(part.lower() in line_lower for part in destination_parts if len(part) > 3):
                # Extract time from format: "XX.Xmin"
                time_match = re.search(r': (\d+\.?\d*)min', line)
                if time_match:
                    travel_time = float(time_match.group(1))
                    return {
                        'travel_time': travel_time,
                        'found_stop': line.strip(),
                        'exact_match': dest_lower in line_lower
                    }
    
    # If no exact match, look for partial matches
    for line in lines:
        if '🚏' in line and ':' in line and 'min' in line:
            line_lower = line.lower()
            
            # Skip origin stops
            if 'emil-figge' in line_lower or 'emil figge' in line_lower:
                continue
            
            # Look for key parts of the destination
            if 'bochum' in dest_lower and 'bochum' in line_lower:
                time_match = re.search(r': (\d+\.?\d*)min', line)
                if time_match:
                    travel_time = float(time_match.group(1))
                    return {
                        'travel_time': travel_time,
                        'found_stop': line.strip(),
                        'exact_match': False
                    }
            
            # Look for street names
            street_parts = destination.split(',')[0].split()
            if len(street_parts) > 1:
                street_name = street_parts[0].lower()
                if len(street_name) > 4 and street_name in line_lower:
                    time_match = re.search(r': (\d+\.?\d*)min', line)
                    if time_match:
                        travel_time = float(time_match.group(1))
                        return {
                            'travel_time': travel_time,
                            'found_stop': line.strip(),
                            'exact_match': False
                        }
    
    return None

def main():
    """Run tests on all samples against DELFI database"""
    print("Testing sample routes against DELFI database...")
    print("=" * 80)
    
    # Read samples
    try:
        with open('samples.txt', 'r') as f:
            samples = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("Error: samples.txt not found")
        return
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        parsed = parse_sample_line(sample)
        if not parsed:
            print(f"Skipping invalid line: {sample}")
            continue
            
        origin, destination, expected_time = parsed
        
        print(f"\nTest {i}: {origin} → {destination}")
        print(f"Expected time: {expected_time} minutes")
        
        # Test with DELFI database
        print("  DELFI database: ", end='', flush=True)
        result = run_delfi_query(origin, destination, max_time=max(60, expected_time + 20))
        print("")  # New line for better formatting
        
        if result:
            travel_time = result['travel_time']
            diff = travel_time - expected_time
            status = "✓" if abs(diff) <= 10 else "✗"  # Allow 10 min tolerance
            match_type = "exact" if result['exact_match'] else "partial"
            print(f"{travel_time:.1f} min (diff: {diff:+.1f}) {status} ({match_type} match)")
            print(f"    Found: {result['found_stop']}")
        else:
            travel_time = None
            print("Destination not found in reachable area")
        
        results.append({
            'origin': origin,
            'destination': destination,
            'expected': expected_time,
            'delfi': travel_time,
            'found_stop': result['found_stop'] if result else None,
            'exact_match': result['exact_match'] if result else False
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - DELFI Database Results:")
    print(f"{'Route':<40} {'Expected':>10} {'DELFI':>10} {'Diff':>8} {'Status':>8}")
    print("-" * 80)
    
    successful_tests = 0
    total_tests = len(results)
    
    for r in results:
        route = f"{r['origin'].split(',')[0]} → {r['destination'].split(',')[0]}"
        if len(route) > 40:
            route = route[:37] + "..."
            
        delfi_str = f"{r['delfi']:.1f}" if r['delfi'] is not None else "N/A"
        
        if r['delfi'] is not None:
            diff = r['delfi'] - r['expected']
            diff_str = f"{diff:+.1f}"
            status = "✓" if abs(diff) <= 10 else "✗"
            if abs(diff) <= 10:
                successful_tests += 1
        else:
            diff_str = "N/A"
            status = "✗"
        
        print(f"{route:<40} {r['expected']:>10} {delfi_str:>10} {diff_str:>8} {status:>8}")
    
    print("-" * 80)
    print(f"Success rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
    
    # Analysis
    print(f"\nAnalysis:")
    found_routes = [r for r in results if r['delfi'] is not None]
    if found_routes:
        avg_diff = sum(abs(r['delfi'] - r['expected']) for r in found_routes) / len(found_routes)
        print(f"  Average absolute difference: {avg_diff:.1f} minutes")
        
        exact_matches = sum(1 for r in results if r['exact_match'])
        print(f"  Exact stop matches: {exact_matches}/{total_tests}")
        
        print(f"  Routes found in DELFI: {len(found_routes)}/{total_tests}")
    else:
        print("  No routes were successfully found in DELFI database")

if __name__ == "__main__":
    main()