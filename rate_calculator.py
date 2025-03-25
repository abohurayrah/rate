import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import os

class BNPLRateCalculator:
    """
    A rate calculator for a B2B BNPL fintech product.
    
    Calculates minimum acceptable monthly interest rates based on:
    - Pipeline pressure (demand)
    - Cash supply (available to issue loans)
    - Instalment duration
    - Minimum IRR requirements
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the calculator with configuration values.
        
        Parameters:
        - config_path: Path to the configuration file. If file doesn't exist,
                      default values will be used and a new config file will be created.
        """
        # Default configuration values
        default_config = {
            # Economic constants
            "cost_of_funds": 0.016,  # 1.6%
            "min_viable_rate": 0.018,  # 1.8%
            "max_rate": 0.032,  # 3.2%
            "fixed_small_loan_rate": 0.04,  # 4% for loans up to 100k SAR
            "small_loan_threshold": 100000,  # 100k SAR
            "min_annual_irr": 0.20,  # 20% minimum annual IRR
            
            # Pipeline pressure buckets (number of leads)
            "pipeline_buckets": {
                "very_low": [0, 10],
                "low": [10, 25],
                "decent": [25, 50],
                "high": [50, 9999]
            },
            
            # Cash supply ranges (in SAR)
            "min_cash": 2000000,  # 2m SAR
            "max_cash": 12000000,  # 12m SAR
            "cash_buckets": 5,  # Number of cash supply divisions
            
            # Instalment durations (months)
            "min_duration": 1,
            "max_duration": 6,
            
            # Rate calculation weights
            "pipeline_pressure_weight": 0.6,  # 60% weight to pipeline pressure
            "cash_supply_weight": 0.4,  # 40% to cash supply
            
            # Duration adjustment factor (0-15% increase for longest duration)
            "max_duration_adjustment": 0.15
        }
        
        # Load configuration from file if it exists, otherwise create it with defaults
        self.config = self._load_or_create_config(config_path, default_config)
        
        # Set instance variables from config
        self.COST_OF_FUNDS = self.config["cost_of_funds"]
        self.MIN_VIABLE_RATE = self.config["min_viable_rate"]
        self.MAX_RATE = self.config["max_rate"]
        self.FIXED_SMALL_LOAN_RATE = self.config["fixed_small_loan_rate"]
        self.SMALL_LOAN_THRESHOLD = self.config["small_loan_threshold"]
        self.MIN_ANNUAL_IRR = self.config["min_annual_irr"]
        
        # Convert pipeline buckets from list to tuple
        self.pipeline_buckets = {
            bucket: tuple(bounds) for bucket, bounds in self.config["pipeline_buckets"].items()
        }
        
        self.min_cash = self.config["min_cash"]
        self.max_cash = self.config["max_cash"]
        self.cash_buckets = self.config["cash_buckets"]
        
        self.min_duration = self.config["min_duration"]
        self.max_duration = self.config["max_duration"]
        
        self.pipeline_pressure_weight = self.config["pipeline_pressure_weight"]
        self.cash_supply_weight = self.config["cash_supply_weight"]
        self.max_duration_adjustment = self.config["max_duration_adjustment"]
    
    def _load_or_create_config(self, config_path, default_config):
        """Load configuration from file or create new config file with defaults."""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Error reading config file {config_path}. Using default values.")
                return default_config
        else:
            # Create config file with default values
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print(f"Created new configuration file at {config_path}")
                return default_config
            except IOError:
                print(f"Could not create config file {config_path}. Using default values.")
                return default_config
    
    def get_pipeline_bucket(self, num_leads):
        """Determine the pipeline pressure bucket based on number of leads."""
        for bucket, (lower, upper) in self.pipeline_buckets.items():
            if lower <= num_leads < upper:
                return bucket
        return "high"  # Default to high if outside defined ranges
    
    def get_cash_bucket_index(self, cash_supply):
        """Determine the index of the cash supply bucket."""
        # Handle exceptionally high cash inflows by capping
        if cash_supply > self.max_cash:
            # If cash is above our normal range, we'll still use our lowest rate
            # but maintain the calculation logic
            cash_supply = min(cash_supply, self.max_cash * 2)
        
        # Normalize cash supply to a 0-1 range relative to min/max cash
        normalized = (cash_supply - self.min_cash) / (self.max_cash - self.min_cash)
        normalized = max(0, min(1, normalized))  # Ensure it's between 0 and 1
        
        # Convert to bucket index
        bucket_index = int(normalized * (self.cash_buckets - 1))
        return bucket_index
    
    def calculate_base_rate(self, pipeline_bucket, cash_bucket_index):
        """Calculate the base rate based on pipeline pressure and cash supply."""
        # Pipeline pressure component - loaded from config
        pipeline_weights = self.config.get("pipeline_weights", {
            "very_low": 0.0,
            "low": 0.3,
            "decent": 0.7,
            "high": 1.0
        })
        pipeline_factor = pipeline_weights.get(pipeline_bucket, 0.0)
        
        # Cash supply component (inverse relationship - lower cash = higher rates)
        # We want a 0-1 range where 0 = max cash (lowest rates) and 1 = min cash (highest rates)
        cash_factor = 1 - (cash_bucket_index / (self.cash_buckets - 1))
        
        # Combine factors with weights from config
        combined_factor = (self.pipeline_pressure_weight * pipeline_factor + 
                           self.cash_supply_weight * cash_factor)
        
        # Scale between minimum viable rate and maximum rate
        base_rate = self.MIN_VIABLE_RATE + combined_factor * (self.MAX_RATE - self.MIN_VIABLE_RATE)
        
        return base_rate
    
    def adjust_for_duration(self, base_rate, duration):
        """Adjust the base rate based on instalment duration."""
        # Normalize duration to a 0-1 scale
        duration_factor = (duration - self.min_duration) / (self.max_duration - self.min_duration)
        
        # Apply duration adjustment using the configured max adjustment factor
        duration_adjustment = self.max_duration_adjustment * duration_factor
        
        # Apply the adjustment
        adjusted_rate = base_rate * (1 + duration_adjustment)
        
        # Ensure the rate doesn't exceed the maximum
        return min(adjusted_rate, self.MAX_RATE)
        
    def calculate_irr(self, rate, duration, loan_amount):
        """
        Calculate the Internal Rate of Return (IRR) for a loan.
        
        Parameters:
        - rate: Monthly interest rate (decimal)
        - duration: Loan duration in months
        - loan_amount: Principal loan amount in SAR
        
        Returns:
        - Annual IRR as a decimal
        """
        # Calculate monthly payment (equal instalments)
        monthly_payment = loan_amount * (rate * (1 + rate)**duration) / ((1 + rate)**duration - 1)
        
        # Create cash flow array (-loan_amount followed by monthly payments)
        cash_flows = [-loan_amount] + [monthly_payment] * duration
        
        # Calculate monthly IRR using numpy's IRR function
        try:
            monthly_irr = np.irr(cash_flows)
            annual_irr = (1 + monthly_irr)**12 - 1
            return annual_irr
        except:
            # Fallback calculation if IRR doesn't converge
            # Simplified IRR approximation: total interest / (avg principal × duration)
            total_interest = monthly_payment * duration - loan_amount
            avg_principal = loan_amount / 2  # Simplified average outstanding principal
            monthly_irr_approx = total_interest / (avg_principal * duration)
            annual_irr_approx = (1 + monthly_irr_approx)**12 - 1
            return annual_irr_approx
    
    def get_min_rate_for_irr(self, duration, loan_amount, target_irr=None):
        """
        Calculate the minimum monthly rate needed to achieve the target IRR.
        
        Parameters:
        - duration: Loan duration in months
        - loan_amount: Principal loan amount in SAR
        - target_irr: Target annual IRR (decimal). If None, uses self.MIN_ANNUAL_IRR
        
        Returns:
        - Minimum monthly rate required to achieve the target IRR
        """
        if target_irr is None:
            target_irr = self.MIN_ANNUAL_IRR
            
        # Convert annual IRR to monthly
        target_monthly_irr = (1 + target_irr)**(1/12) - 1
        
        # Start with a reasonable guess - slightly above the target monthly IRR
        rate_guess = target_monthly_irr * 1.1
        
        # Binary search to find the rate that produces the target IRR
        min_rate = self.MIN_VIABLE_RATE
        max_rate = self.MAX_RATE
        rate = rate_guess
        tolerance = 0.0001
        
        for _ in range(20):  # Limit iterations to prevent infinite loops
            calculated_irr = self.calculate_irr(rate, duration, loan_amount)
            
            if abs(calculated_irr - target_irr) < tolerance:
                break
                
            if calculated_irr < target_irr:
                min_rate = rate
                rate = (rate + max_rate) / 2
            else:
                max_rate = rate
                rate = (min_rate + rate) / 2
        
        return max(rate, self.MIN_VIABLE_RATE)
    
    def get_rate(self, loan_amount, num_leads, cash_supply, duration):
        """
        Calculate the appropriate interest rate for a loan.
        
        Parameters:
        - loan_amount: Amount of the loan in SAR
        - num_leads: Number of leads in the pipeline (measure of demand)
        - cash_supply: Available cash to issue loans in SAR
        - duration: Instalment duration in months (1-6)
        
        Returns:
        - Monthly interest rate as a decimal
        """
        # Apply fixed rate for small loans
        if loan_amount <= self.SMALL_LOAN_THRESHOLD:
            return self.FIXED_SMALL_LOAN_RATE
        
        # Get pipeline pressure bucket
        pipeline_bucket = self.get_pipeline_bucket(num_leads)
        
        # Get cash supply bucket index
        cash_bucket_index = self.get_cash_bucket_index(cash_supply)
        
        # Calculate base rate from pipeline and cash factors
        base_rate = self.calculate_base_rate(pipeline_bucket, cash_bucket_index)
        
        # Adjust for duration
        adjusted_rate = self.adjust_for_duration(base_rate, duration)
        
        # Calculate the minimum rate needed to achieve target IRR
        irr_based_rate = self.get_min_rate_for_irr(duration, loan_amount)
        
        # Take the maximum of the adjusted rate and the IRR-based minimum rate
        final_rate = max(adjusted_rate, irr_based_rate, self.MIN_VIABLE_RATE)
        
        # Ensure the rate doesn't exceed the maximum rate
        return min(final_rate, self.MAX_RATE)
    
    def generate_rate_matrix(self, loan_amount, duration=3):
        """
        Generate a matrix of rates for different pipeline pressures and cash supplies.
        
        Parameters:
        - loan_amount: Amount of the loan in SAR
        - duration: Instalment duration in months (default: 3)
        
        Returns:
        - pandas DataFrame with rates for different combinations
        """
        # For small loans, return a simple matrix with fixed rate
        if loan_amount <= self.SMALL_LOAN_THRESHOLD:
            pipeline_buckets = list(self.pipeline_buckets.keys())
            cash_ranges = [f"{i}M-{i+2}M" for i in range(2, 12, 2)]
            
            # Create a matrix filled with the fixed rate
            rate_matrix = pd.DataFrame(
                data=[[self.FIXED_SMALL_LOAN_RATE for _ in cash_ranges] for _ in pipeline_buckets],
                index=pipeline_buckets,
                columns=cash_ranges
            )
            return rate_matrix
        
        # For larger loans, calculate the dynamic matrix
        pipeline_buckets = list(self.pipeline_buckets.keys())
        
        # Create cash supply ranges
        cash_values = np.linspace(self.min_cash, self.max_cash, self.cash_buckets)
        cash_ranges = [f"{int(cash_values[i]/1_000_000)}M-{int(cash_values[i+1]/1_000_000)}M" 
                      for i in range(len(cash_values)-1)]
        
        # Representative values for calculations
        pipeline_representatives = {
            "very_low": 5,
            "low": 15,
            "decent": 35,
            "high": 75
        }
        
        # Create the rate matrix
        rate_matrix = []
        for bucket in pipeline_buckets:
            row = []
            for i in range(len(cash_values)-1):
                # Use midpoint of cash range
                cash_mid = (cash_values[i] + cash_values[i+1]) / 2
                rate = self.get_rate(loan_amount, pipeline_representatives[bucket], cash_mid, duration)
                row.append(rate)
            rate_matrix.append(row)
        
        # Convert to pandas DataFrame
        rate_df = pd.DataFrame(
            data=rate_matrix,
            index=pipeline_buckets,
            columns=cash_ranges
        )
        
        return rate_df
    
    def generate_duration_matrices(self, loan_amount):
        """
        Generate rate matrices for different instalment durations.
        
        Parameters:
        - loan_amount: Amount of the loan in SAR
        
        Returns:
        - Dictionary of DataFrames, one for each duration
        """
        matrices = {}
        for duration in range(self.min_duration, self.max_duration + 1):
            matrices[duration] = self.generate_rate_matrix(loan_amount, duration)
        return matrices
    
    def visualize_rate_matrix(self, rate_matrix, duration=None):
        """Create a heatmap visualization of the rate matrix."""
        plt.figure(figsize=(10, 8))
        
        # Format the values as percentages (using DataFrame.map instead of applymap)
        formatted_matrix = rate_matrix.copy()
        for col in formatted_matrix.columns:
            formatted_matrix[col] = formatted_matrix[col].map(lambda x: f"{x:.1%}")
        
        # Create a heatmap of the numeric values
        ax = sns.heatmap(rate_matrix, annot=formatted_matrix, fmt="", cmap="YlOrRd", 
                         linewidths=0.5, cbar_kws={'label': 'Monthly Interest Rate'})
        
        title = "BNPL Monthly Interest Rate Matrix"
        if duration is not None:
            title += f" - {duration} Month Duration"
        
        plt.title(title)
        plt.xlabel("Cash Supply (SAR)")
        plt.ylabel("Pipeline Pressure")
        plt.tight_layout()
        
        return plt

# Example usage
if __name__ == "__main__":
    calculator = BNPLRateCalculator()
    
    print("Example: Rate for a 200,000 SAR loan with decent pipeline pressure (35 leads),")
    print("8M SAR cash supply, and 3-month duration:")
    rate = calculator.get_rate(200000, 35, 8_000_000, 3)
    print(f"Calculated rate: {rate:.2%}\n")
    
    print("Rate Matrix for 150,000 SAR loan with 3-month duration:")
    rate_matrix = calculator.generate_rate_matrix(150000, 3)
    
    # Format and print the matrix (using df.copy and map instead of applymap)
    formatted_matrix = rate_matrix.copy()
    for col in formatted_matrix.columns:
        formatted_matrix[col] = formatted_matrix[col].map(lambda x: f"{x:.2%}")
    print(formatted_matrix)
    
    # Visualize the matrix
    plt.figure(figsize=(10, 8))
    calculator.visualize_rate_matrix(rate_matrix, 3)
    plt.savefig("rate_matrix_3month.png")
    
    print("\nRate Matrix for 80,000 SAR loan (fixed rate):")
    small_loan_matrix = calculator.generate_rate_matrix(80000)
    
    # Format and print the small loan matrix
    formatted_small_matrix = small_loan_matrix.copy()
    for col in formatted_small_matrix.columns:
        formatted_small_matrix[col] = formatted_small_matrix[col].map(lambda x: f"{x:.2%}")
    print(formatted_small_matrix) 