# FlashDNA Analysis Flow Fix

This package contains diagnostic tools and patches to fix critical issues in the FlashDNA analysis flow, particularly focusing on:

1. **Runway calculation errors** causing extreme values (9,999 months)
2. **Missing data validation** leading to UI display problems
3. **Error handling issues** in model predictions and calculations
4. **Fallbacks for missing values** to ensure a consistent user experience

## Quick Start

To quickly fix the issues in your FlashDNA system:

```bash
# Dry run mode (shows what would be fixed but doesn't modify files)
python flashdna_flow_fix.py

# Apply fixes to the system
python flashdna_flow_fix.py --apply
```

## Files Included

- **flashdna_flow_fix.py**: Main script that applies all fixes
- **analysis_flow_diagnostics.py**: Diagnostics tool for identifying and fixing issues in analysis documents
- **scenario_runway_fix.py**: Fixed implementation of runway calculation to prevent division by zero/small numbers
- **run_analysis_fix.py**: Enhanced run_analysis function with proper error handling and validation

## How It Works

The fix package addresses issues in several key areas:

### 1. Runway Calculation Fixes

The `scenario_runway` function in the original code has issues with:
- Division by zero when `burn_rate` is 0
- Extremely small divisors when `burn_rate` and `monthly_revenue` are nearly equal
- No upper bound on runway values

Our fix applies a more robust implementation:
- Ensures burn rate is always positive and reasonable
- Sets a minimum net burn to prevent division by very small numbers
- Caps runway at a reasonable maximum (120 months)
- Handles the case where revenue exceeds burn rate correctly

### 2. Data Validation and Required Fields

The system fails to properly validate input data and ensure all required fields exist:

- Our fix adds comprehensive validation of required fields
- Ensures all CAMP scores (capital, market, advantage, people) have valid values
- Provides intelligent defaults based on available metrics
- Prevents NaN, None, and out-of-range values

### 3. Error Handling

The original code has insufficient error handling, causing UI issues when calculations fail:

- Enhanced error boundaries around critical calculation sections
- Graceful fallbacks when model loading or predictions fail
- Better logging of errors for diagnosis
- Prevention of cascade failures where one error affects multiple UI components

## Usage Details

### Fixing the Codebase

```bash
# Apply all fixes and create backups of original files
python flashdna_flow_fix.py --apply

# Apply fixes without creating backups
python flashdna_flow_fix.py --apply --no-backup
```

### Diagnosing Documents

You can also diagnose and fix issues in specific analysis documents:

```bash
# Analyze a document without modifying it
python flashdna_flow_fix.py --doc path/to/document.json

# Analyze and fix document issues
python flashdna_flow_fix.py --apply --doc path/to/document.json
```

The fixed document will be saved as `path/to/document.json.fixed.json`.

## Implementation Details

### Running Diagnostics Programmatically

You can use the diagnostics tool in your code:

```python
from analysis_flow_diagnostics import FlashDnaDiagnostics

# Create a diagnostics instance
diagnostics = FlashDnaDiagnostics()

# Check a document for issues
issues = diagnostics.diagnose_all()

# Fix issues in the document
fixed_doc = diagnostics.fix_all()
```

### Using the Safe Runway Calculation

If you only need the fixed runway calculation:

```python
from scenario_runway_fix import safe_scenario_runway

# Calculate runway with safety checks
runway, ending_cash, cash_flow = safe_scenario_runway(
    burn_rate=30000,
    current_cash=500000,
    monthly_revenue=20000,
    rev_growth=0.1,
    cost_growth=0.05
)
```

## After Applying Fixes

After applying the fixes:

1. Restart your FlashDNA application
2. If using Streamlit, clear the cache: `streamlit cache clear`
3. Test the fixes by analyzing a startup with parameters that previously caused issues

## Troubleshooting

If you encounter issues after applying the fixes:

1. Check the log files for errors
2. Restore from backups if needed (files are backed up with `.backup` extension)
3. Try applying individual fixes instead of all at once

## Support

If you have questions or need further assistance with these fixes, please contact the FlashDNA support team. 