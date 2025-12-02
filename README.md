# Battery CFD Automation Tool  
Automated Post-Processing Framework for Battery Cooling CFD Simulations

This tool automates all post-processing steps for CFD simulations exported as CSV files.  
It was developed to support Battery Thermal Management studies by eliminating manual inspection and enabling fast, repeatable, quantitative comparison across cases (different inlet velocities, heat loads, geometries, etc.).

---

## Key Features

### 1. Automatic CSV Discovery
- Recursively scans a directory for CSV files.
- Allows custom filename patterns (e.g., `batt*.csv`).

### 2. Automatic Numeric Column Detection
- No hardcoded variable names.
- Works with any CFD output (Autodesk CFD, Fluent, StarCCM+, OpenFOAM CSV exports).

### 3. Extended Statistics for All Fields
For each numeric field in each file:
- `count`
- `min`, `max`
- `mean`
- `std` (standard deviation)
- `p05`, `p50`, `p95` (percentile values)

### 4. Metadata Extraction from Filenames
Example:  
`batt 12 m per sec.csv` → inlet velocity parsed as **12.0 m/s**

Stored in the summary file as:
- `case_name`
- `inlet_velocity_mps`

### 5. Per-File Histograms
- One histogram PNG per numeric field.
- Reveals distributions of temperature, velocity, turbulence quantities, etc.

### 6. Cross-Case Comparison Plots
For any selected field (e.g. Temp, Vy Vel, Pressure), the tool generates:
- Bar charts of **mean values** across all cases.

### 7. Consolidated Summary CSV
Saved as:
