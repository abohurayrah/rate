# BNPL Rate Calculator Matrix

A Python-based rate calculator matrix for B2B BNPL (Buy Now, Pay Later) fintech products. This tool outputs minimum acceptable monthly interest rates under varying market conditions, using a rational economic model.

## Overview

The calculator determines appropriate interest rates based on three key inputs:
1. **Pipeline Pressure** - Demand for loans measured by number of leads
2. **Cash Supply** - Available capital to issue loans
3. **Instalment Duration** - Length of the loan term in months

## Features

- Fixed 4% monthly rate for loans up to 100k SAR
- Dynamic rate calculation for loans over 100k SAR (2% - 3.5% range)
- Economic model that accounts for:
  - Higher rates during high demand periods
  - Higher rates when cash reserves are lower
  - Rate adjustments for longer instalment durations
- Visualization capabilities (heatmaps of rate matrices)
- Ability to handle exceptional cash inflows
- **NEW:** Interactive dashboard with dynamic updates
- **NEW:** Sensitivity analysis and what-if scenario testing
- **NEW:** Monte Carlo simulation for rate variability assessment

## Economic Model

The model is built on these principles:

- **Cost of Funds**: ~1.6%
- **Minimum Viable Rate**: 1.8% (economic floor)
- **Maximum Rate**: 3.5% (for high pipeline pressure)

The final rate is calculated through:
1. Determining the base rate from pipeline pressure and cash supply
2. Adjusting the base rate for instalment duration
3. Ensuring the rate stays within viable economic boundaries

## Usage

### Basic Python API

```python
from rate_calculator import BNPLRateCalculator

# Create calculator instance
calculator = BNPLRateCalculator()

# Get rate for a specific scenario
rate = calculator.get_rate(
    loan_amount=200000,  # 200k SAR
    num_leads=35,        # 35 leads (decent pressure)
    cash_supply=8000000, # 8M SAR cash available
    duration=3           # 3-month instalment plan
)
print(f"Calculated rate: {rate:.2%}")

# Generate a rate matrix for a specific loan amount and duration
rate_matrix = calculator.generate_rate_matrix(loan_amount=150000, duration=3)
print(rate_matrix)

# Visualize the rate matrix
calculator.visualize_rate_matrix(rate_matrix, duration=3)
```

### Interactive Dashboard

The calculator now includes an interactive Streamlit dashboard:

```bash
# Run the interactive dashboard
streamlit run app.py
```

The dashboard includes:

- **Rate Calculator**: Adjust parameters via sliders and see real-time rate updates
- **Sensitivity Analysis**: Understand how changes in each parameter affect the calculated rate
- **Scenario Testing**: Run simulations to analyze rate variability under different conditions

## Interactive Features

### Data Input & Dynamic Updates
- Input key parameters such as cash on hand, instalment duration, and pipeline pressure
- Real-time matrix recalculation as inputs change

### Sensitivity Analysis & Scenario Testing
- Test "what-if" scenarios to understand parameter impact
- Monte Carlo simulation to assess rate variability
- Visualization of rate sensitivity to different inputs

### User Interface
- Clean, intuitive dashboard for easy parameter manipulation
- Interactive visualizations including heatmaps and line charts
- Detailed tooltips and documentation for better understanding

## Customization

The calculator is fully parameterizable. You can adjust:

- Pipeline pressure bucket definitions
- Cash supply ranges
- Economic constants (cost of funds, minimum viable rate, etc.)
- Weighting between different factors
- Duration adjustment formula

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit
- Plotly 