import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BNPLRateCalculator:
    """
    A rate calculator for a B2B BNPL fintech product.
    
    Calculates minimum acceptable monthly interest rates based on:
    - Pipeline pressure (demand)
    - Cash supply (available to issue loans)
    - Instalment duration
    """
    
    def __init__(self):
        # Economic constants
        self.COST_OF_FUNDS = 0.016  # 1.6%
        self.MIN_VIABLE_RATE = 0.018  # 1.8%
        self.MAX_RATE = 0.035  # 3.5%
        self.FIXED_SMALL_LOAN_RATE = 0.04  # 4% for loans up to 100k SAR
        self.SMALL_LOAN_THRESHOLD = 100000  # 100k SAR
        
        # Pipeline pressure buckets (number of leads)
        self.pipeline_buckets = {
            "very_low": (0, 10),
            "low": (10, 25),
            "decent": (25, 50),
            "high": (50, float('inf'))
        }
        
        # Cash supply ranges (in SAR)
        self.min_cash = 2_000_000  # 2m SAR
        self.max_cash = 12_000_000  # 12m SAR
        self.cash_buckets = 5  # Number of cash supply divisions
        
        # Instalment durations (months)
        self.min_duration = 1
        self.max_duration = 6
    
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
        # Pipeline pressure component
        pipeline_weights = {
            "very_low": 0.0,
            "low": 0.3,
            "decent": 0.7,
            "high": 1.0
        }
        pipeline_factor = pipeline_weights[pipeline_bucket]
        
        # Cash supply component (inverse relationship - lower cash = higher rates)
        # We want a 0-1 range where 0 = max cash (lowest rates) and 1 = min cash (highest rates)
        cash_factor = 1 - (cash_bucket_index / (self.cash_buckets - 1))
        
        # Combine factors with appropriate weights
        # 60% weight to pipeline pressure, 40% to cash supply
        combined_factor = 0.6 * pipeline_factor + 0.4 * cash_factor
        
        # Scale between minimum viable rate and maximum rate
        base_rate = self.MIN_VIABLE_RATE + combined_factor * (self.MAX_RATE - self.MIN_VIABLE_RATE)
        
        return base_rate
    
    def adjust_for_duration(self, base_rate, duration):
        """Adjust the base rate based on instalment duration."""
        # Normalize duration to a 0-1 scale
        duration_factor = (duration - self.min_duration) / (self.max_duration - self.min_duration)
        
        # Apply a smaller adjustment for duration (0-15% increase for longest duration)
        duration_adjustment = 0.15 * duration_factor
        
        # Apply the adjustment
        adjusted_rate = base_rate * (1 + duration_adjustment)
        
        # Ensure the rate doesn't exceed the maximum
        return min(adjusted_rate, self.MAX_RATE)
    
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
        final_rate = self.adjust_for_duration(base_rate, duration)
        
        # Ensure the rate doesn't fall below the minimum viable rate
        return max(final_rate, self.MIN_VIABLE_RATE)
    
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