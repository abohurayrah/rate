# BNPL Rate Calculator Configuration

This document explains the configuration parameters for the BNPL Rate Calculator.

## Configuration Parameters

The calculator's behavior can be customized by editing the `config.json` file.

### Economic Constants

- `cost_of_funds` (0.016): The cost of funding loans, currently set at 1.6%.
- `min_viable_rate` (0.018): The absolute minimum viable monthly profit rate (1.8%).
- `max_rate` (0.032): The maximum monthly profit rate that can be charged (3.2%).
- `fixed_small_loan_rate` (0.04): The fixed profit rate for small loans (4.0%).
- `small_loan_threshold` (100000): The threshold below which loans use the fixed rate (100,000 SAR).
- `min_annual_irr` (0.20): The minimum annual Internal Rate of Return target (20%).

### Pipeline Pressure Settings

- `pipeline_buckets`: Defines the ranges for pipeline pressure categories.
  - `very_low`: 0-9 leads
  - `low`: 10-24 leads
  - `decent`: 25-49 leads
  - `high`: 50+ leads

- `pipeline_weights`: The weight factors for each pipeline pressure category.
  - `very_low`: 0.0 (minimum pressure)
  - `low`: 0.3
  - `decent`: 0.7
  - `high`: 1.0 (maximum pressure)

### Cash Supply Settings

- `min_cash` (2,000,000): Minimum cash supply threshold in SAR.
- `max_cash` (12,000,000): Maximum cash supply threshold in SAR.
- `cash_buckets` (5): Number of divisions for cash supply ranges.

### Duration Settings

- `min_duration` (1): Minimum loan duration in months.
- `max_duration` (6): Maximum loan duration in months.

### Rate Calculation Weights

- `pipeline_pressure_weight` (0.6): Weight given to pipeline pressure in base rate calculation (60%).
- `cash_supply_weight` (0.4): Weight given to cash supply in base rate calculation (40%).
- `max_duration_adjustment` (0.15): Maximum adjustment for duration (15% increase for longest duration).

## Modifying Configuration

To modify the configuration:

1. Open `config.json` in a text editor
2. Change the desired parameters
3. Save the file
4. Restart the application for changes to take effect

Note: Keep the JSON format intact to avoid parsing errors.