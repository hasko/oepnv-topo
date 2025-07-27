# GitHub Issues to Create

## Issue 1: Bug - End walking time incorrectly added to total travel duration

**Labels**: bug, high-priority

**Description:**
In the `add_end_walking_expansion()` function, the final walking time is incorrectly added to the total travel time, causing points to exceed the specified time limit.

**Problem:**
Line 402 in `main.py`:
```python
total_time_minutes = stop_info['total_time_minutes'] + walk_time_minutes
```

This violates the constraint that the maximum travel duration should already include both initial and final walk times.

**Expected Behavior:**
The algorithm should:
1. Respect the original time budget (e.g., 30 minutes)
2. Only include walking destinations where `transit_time + final_walk_time ≤ max_time`
3. Not add walking time on top of the time limit

**Impact:**
- Points that should be reachable within the time limit are being excluded
- The isochrone boundaries are smaller than they should be
- Users get incomplete results for reachable areas

**Fix Applied:**
Modified `add_end_walking_expansion()` to:
- Add `max_total_time_minutes` parameter
- Filter walking destinations by total time constraint
- Only include destinations that fit within the original time budget

**Test Case:**
From Düsseldorf Hauptbahnhof with 20-minute limit:
- **Before fix**: Points at 18min transit + 5min walk = 23min were excluded
- **After fix**: Points should be included if total ≤ 20min

**Files Changed:**
- `main.py`: `add_end_walking_expansion()` function and its callers

---

## Issue 2: Missing bus line 447/440 connection in Dortmund area

**Labels**: bug, investigation-needed

**Description:**
Known 30-minute connection between two addresses in Dortmund using bus lines 447 or 440 is not being found by the isochrone calculation.

**Test Case:**
- **Start**: University area in Dortmund (near Dortmund Universität S stop)
- **End**: Residential area in Dortmund (near Dortmund Am Hombruchsfeld stop)
- **Expected**: Reachable within 30 minutes via bus line 447 or 440
- **Actual**: Destination appears outside the 60-minute isochrone

**Geocoding Results:**
- Address A: ✅ Coordinates: 51.492x, 7.416x (closest stop: Dortmund Universität S - 0.17km)
- Address B: ✅ Coordinates: 51.472x, 7.448x (closest stop: Dortmund Am Hombruchsfeld - 0.11km)

**GTFS Data Verification:**
✅ Bus lines found in database:
- Line 447: Route ID `de:vrr:447:DSW-38-47` (Hacheney - Renninghausen)
- Line 440: Route ID `de:vrr:440:DSW-38-40` (DO-Lütgendortmund/Germania - DO-Aplerbeck)

**Potential Causes:**
1. **Missing stop connections**: The route between Universität S and Am Hombruchsfeld might not be properly connected
2. **Graph filtering**: Geographic filtering might exclude relevant intermediate stops
3. **Schedule data**: Missing or incomplete schedule information for these lines
4. **Transfer connections**: Required transfers might not be properly modeled

**Investigation Needed:**
1. Check if lines 447/440 actually serve both origin and destination areas
2. Verify stop sequences and schedules in GTFS data
3. Test with broader geographic filters
4. Examine intermediate stops and required transfers

**Debug Commands:**
```bash
# Check available routes from origin area
uv run python main.py debug --route "447"
uv run python main.py debug --route "440"

# Test the connection
uv run python main.py debug --address1 "[ADDRESS_A]" --address2 "[ADDRESS_B]"

# Test isochrone calculation
uv run python main.py query --address "[ADDRESS_A]" --time 60
```

**Additional Context:**
There's also reportedly a second connection using bus line 440 with some walking involved.