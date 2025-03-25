import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import os

class BNPLRateCalculator:
    """
    A rate calculator for a B2B BNPL fintech product.
    
    Calculates minimum acceptable monthly profit rates based on:
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
            "admin_fee_rate": 0.015,  # 1.5% admin fee
            
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
            "max_duration_adjustment": 0.15,
            
            # Repayment structures
            "repayment_structures": {
                "monthly": {
                    "display_name": "Monthly EMI",
                    "description": "Equal monthly installments paid every month"
                },
                "bullet": {
                    "display_name": "Bullet Payment",
                    "description": "Single payment at the end of the loan term"
                },
                "bi_monthly": {
                    "display_name": "Bi-Monthly EMI",
                    "description": "Equal installments paid every 2 months"
                },
                "quarterly": {
                    "display_name": "Quarterly EMI",
                    "description": "Equal installments paid every 3 months"
                }
            }
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
        self.ADMIN_FEE_RATE = self.config.get("admin_fee_rate", 0.015)  # Default to 1.5% if not present
        
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
        
        # Repayment structures
        self.repayment_structures = self.config.get("repayment_structures", {
            "monthly": {"display_name": "Monthly EMI"},
            "bullet": {"display_name": "Bullet Payment"},
            "bi_monthly": {"display_name": "Bi-Monthly EMI"},
            "quarterly": {"display_name": "Quarterly EMI"}
        })
    
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
        
    def get_cashflows(self, rate, duration, loan_amount, repayment_structure='monthly', include_admin_fee=True):
        """
        Calculate the cashflow sequence for a loan with different repayment structures.
        
        Parameters:
        - rate: Monthly profit rate (decimal)
        - duration: Loan duration in months
        - loan_amount: Principal loan amount in SAR
        - repayment_structure: Type of repayment structure
        - include_admin_fee: Whether to include admin fee in calculation
        
        Returns:
        - List of cash flows (starting with the loan outflow)
        """
        # Calculate total profit and admin fee
        total_profit = loan_amount * rate * duration
        admin_fee = loan_amount * self.ADMIN_FEE_RATE if include_admin_fee else 0
        total_repayment = loan_amount + total_profit
        
        # Initial outflow is loan amount minus admin fee (if fee is collected upfront)
        initial_outflow = -loan_amount
        if include_admin_fee:
            # Admin fee is collected upfront, reducing the initial outflow
            initial_outflow += admin_fee
        
        # Initialize cash flow array with initial outflow
        cash_flows = [initial_outflow]
        
        # Calculate payments based on repayment structure
        if repayment_structure == 'bullet':
            # Single payment at the end
            cash_flows = [initial_outflow] + [0] * (duration - 1) + [total_repayment]
            
        elif repayment_structure == 'monthly':
            # Monthly equal payments using standard EMI formula
            if rate > 0:
                # Handle case when rate is very close to zero to avoid division by zero
                if abs(rate) < 1e-10:
                    monthly_payment = loan_amount / duration
                else:
                    monthly_payment = loan_amount * (rate * (1 + rate)**duration) / ((1 + rate)**duration - 1)
            else:
                # If rate is 0, simple division
                monthly_payment = loan_amount / duration
                
            cash_flows = [initial_outflow] + [monthly_payment] * duration
            
        elif repayment_structure == 'bi_monthly':
            # Bi-monthly payments (every 2 months)
            if duration % 2 != 0:
                # If duration is odd, adjust for final payment
                n_payments = duration // 2
                if rate > 0:
                    # Special handling for very small rates to avoid division by zero
                    if abs(rate) < 1e-6:
                        bi_monthly_payment = loan_amount / (n_payments + 0.5)
                    else:
                        # Calculate bi-monthly rate for EMI formula
                        bi_monthly_rate = (1 + rate)**2 - 1
                        
                        # Handle case when bi-monthly rate is very close to zero 
                        if abs(bi_monthly_rate) < 1e-6 or abs((1 + bi_monthly_rate)**n_payments - 1) < 1e-6:
                            bi_monthly_payment = loan_amount / (n_payments + 0.5)
                        else:
                            # Calculate payment for n payments + 1 final payment
                            bi_monthly_payment = loan_amount * (bi_monthly_rate * (1 + bi_monthly_rate)**n_payments) / ((1 + bi_monthly_rate)**n_payments - 1)
                    
                    # Calculate what's left after all full bi-monthly payments
                    if abs(rate) < 1e-6:
                        # When rate is close to zero
                        remaining_principal = loan_amount - n_payments * bi_monthly_payment
                    else:
                        # Only calculate using bi_monthly_rate if it was defined
                        if 'bi_monthly_rate' in locals():
                            remaining_principal = loan_amount - n_payments * (bi_monthly_payment - loan_amount * bi_monthly_rate)
                        else:
                            remaining_principal = loan_amount - n_payments * bi_monthly_payment
                    
                    # Calculate final month's payment
                    if abs(rate) < 1e-6:
                        last_payment = remaining_principal
                    else:
                        last_payment = remaining_principal * (1 + rate)
                else:
                    # If rate is 0, simpler calculation
                    bi_monthly_payment = (loan_amount / duration) * 2
                    last_payment = loan_amount / duration
                
                cash_flows = [initial_outflow]
                for i in range(n_payments):
                    cash_flows.extend([0, bi_monthly_payment])
                cash_flows.append(last_payment)
            else:
                # If duration is even, equal payments every 2 months
                n_payments = duration // 2
                if rate > 0:
                    # Special handling for very small rates to avoid division by zero
                    if abs(rate) < 1e-6:
                        bi_monthly_payment = loan_amount / n_payments
                    else:
                        # Calculate bi-monthly rate for EMI formula
                        bi_monthly_rate = (1 + rate)**2 - 1
                        
                        # Handle case when bi-monthly rate is very close to zero
                        if abs(bi_monthly_rate) < 1e-6 or abs((1 + bi_monthly_rate)**n_payments - 1) < 1e-6:
                            bi_monthly_payment = loan_amount / n_payments
                        else:
                            bi_monthly_payment = loan_amount * (bi_monthly_rate * (1 + bi_monthly_rate)**n_payments) / ((1 + bi_monthly_rate)**n_payments - 1)
                else:
                    # If rate is 0, simple division
                    bi_monthly_payment = loan_amount / n_payments
                
                cash_flows = [initial_outflow]
                for i in range(n_payments):
                    cash_flows.extend([0, bi_monthly_payment])
            
        elif repayment_structure == 'quarterly':
            # Quarterly payments (every 3 months)
            n_payments = duration // 3
            remainder = duration % 3
            
            if remainder == 0:
                # Duration is divisible by 3, equal quarterly payments
                if rate > 0:
                    # Special handling for very small rates to avoid division by zero
                    if abs(rate) < 1e-6:
                        quarterly_payment = loan_amount / n_payments
                    else:
                        # Calculate quarterly rate for EMI formula
                        quarterly_rate = (1 + rate)**3 - 1
                        
                        # Handle case when quarterly rate is very close to zero
                        if abs(quarterly_rate) < 1e-6 or abs((1 + quarterly_rate)**n_payments - 1) < 1e-6:
                            quarterly_payment = loan_amount / n_payments
                        else:
                            quarterly_payment = loan_amount * (quarterly_rate * (1 + quarterly_rate)**n_payments) / ((1 + quarterly_rate)**n_payments - 1)
                else:
                    # If rate is 0, simple division
                    quarterly_payment = loan_amount / n_payments
                
                cash_flows = [initial_outflow]
                for i in range(n_payments):
                    cash_flows.extend([0, 0, quarterly_payment])
            else:
                # Handle remainder months with separate calculation
                if rate > 0:
                    # Special handling for very small rates to avoid division by zero
                    if abs(rate) < 1e-6:
                        quarterly_payment = loan_amount / (n_payments + remainder/3)
                    else:
                        # Calculate quarterly rate for EMI formula
                        quarterly_rate = (1 + rate)**3 - 1
                        
                        # Handle case when quarterly rate is very close to zero
                        if abs(quarterly_rate) < 1e-6 or abs((1 + quarterly_rate)**n_payments - 1) < 1e-6:
                            quarterly_payment = loan_amount / (n_payments + remainder/3)
                        else:
                            quarterly_payment = loan_amount * (quarterly_rate * (1 + quarterly_rate)**n_payments) / ((1 + quarterly_rate)**n_payments - 1)
                    
                    # Calculate what's left after all full quarterly payments
                    if abs(rate) < 1e-6:
                        # When rate is close to zero
                        remaining_principal = loan_amount - n_payments * quarterly_payment
                    else:
                        # Only calculate using quarterly_rate if it was defined
                        if 'quarterly_rate' in locals():
                            remaining_principal = loan_amount - n_payments * (quarterly_payment - loan_amount * quarterly_rate)
                        else:
                            remaining_principal = loan_amount - n_payments * quarterly_payment
                    
                    # Calculate final payment with remaining months of interest
                    if abs(rate) < 1e-6:
                        last_payment = remaining_principal
                    else:
                        last_payment = remaining_principal * (1 + rate)**remainder
                else:
                    # If rate is 0, simpler calculation
                    quarterly_payment = (loan_amount / duration) * 3
                    last_payment = (loan_amount / duration) * remainder
                
                cash_flows = [initial_outflow]
                for i in range(n_payments):
                    cash_flows.extend([0, 0, quarterly_payment])
                
                # Add zeros and final payment based on remainder
                if remainder == 1:
                    cash_flows.append(last_payment)
                elif remainder == 2:
                    cash_flows.extend([0, last_payment])
        
        return cash_flows
    
    def calculate_irr(self, rate, duration, loan_amount, repayment_structure='monthly', include_admin_fee=True):
        """
        Calculate the Internal Rate of Return (IRR) for a loan with different repayment structures.
        
        Parameters:
        - rate: Monthly profit rate (decimal)
        - duration: Loan duration in months
        - loan_amount: Principal loan amount in SAR
        - repayment_structure: Type of repayment structure. Options:
            - 'monthly': Monthly equal payments (default)
            - 'bullet': Single payment at the end of the term
            - 'bi_monthly': Payment every 2 months
            - 'quarterly': Payment every 3 months
        - include_admin_fee: Whether to include admin fee in calculation
        
        Returns:
        - Annual IRR as a decimal
        """
        # Calculate Admin Fee
        admin_fee = loan_amount * self.ADMIN_FEE_RATE if include_admin_fee else 0
        
        # Create a simple cashflow list based on repayment structure
        if repayment_structure == 'bullet':
            # Bullet payment - single payment at end
            # Initial outflow: -loan_amount (admin fee is collected upfront)
            initial_outflow = -loan_amount + admin_fee
            # Single inflow at the end = principal + profit
            total_profit = loan_amount * rate * duration  # Simple interest calculation
            final_payment = loan_amount + total_profit
            
            # Cashflow: initial outflow followed by zeros and final payment
            cash_flows = [initial_outflow] + [0] * (duration - 1) + [final_payment]
            
        elif repayment_structure == 'monthly':
            # Monthly EMI payments
            # Initial outflow (with admin fee collected upfront)
            initial_outflow = -loan_amount + admin_fee
            
            # Calculate EMI (Equal Monthly Installment)
            monthly_payment = loan_amount * (1 + rate * duration) / duration
            
            # Cashflow: initial outflow followed by monthly payments
            cash_flows = [initial_outflow] + [monthly_payment] * duration
            
        elif repayment_structure == 'bi_monthly':
            # Bi-monthly payments
            # Initial outflow (with admin fee collected upfront)
            initial_outflow = -loan_amount + admin_fee
            
            n_payments = duration // 2
            remainder = duration % 2
            
            # Calculate bi-monthly payment
            bi_monthly_payment = loan_amount * (1 + rate * duration) / (n_payments + remainder/2)
            
            # Construct cashflow
            cash_flows = [initial_outflow]
            for i in range(n_payments):
                cash_flows.extend([0, bi_monthly_payment])
            
            # Add final payment if duration is odd
            if remainder:
                cash_flows.append(bi_monthly_payment * remainder/2)
                
        elif repayment_structure == 'quarterly':
            # Quarterly payments
            # Initial outflow (with admin fee collected upfront)
            initial_outflow = -loan_amount + admin_fee
            
            n_payments = duration // 3
            remainder = duration % 3
            
            # Calculate quarterly payment
            quarterly_payment = loan_amount * (1 + rate * duration) / (n_payments + remainder/3)
            
            # Construct cashflow
            cash_flows = [initial_outflow]
            for i in range(n_payments):
                cash_flows.extend([0, 0, quarterly_payment])
            
            # Add final payment if there's a remainder
            if remainder == 1:
                cash_flows.append(quarterly_payment * 1/3)
            elif remainder == 2:
                cash_flows.extend([0, quarterly_payment * 2/3])
        
        # Calculate IRR from cashflows using numpy
        try:
            monthly_irr = np.irr(cash_flows)
            annual_irr = (1 + monthly_irr)**12 - 1
            return annual_irr
        except:
            # Simplified alternative calculation if IRR fails
            total_profit = sum(cash_flows) + loan_amount - admin_fee
            annual_irr = (rate * 12) * (1 + self.ADMIN_FEE_RATE/duration) 
            return annual_irr
    
    def get_min_rate_for_irr(self, duration, loan_amount, target_irr=None, repayment_structure='monthly'):
        """
        Calculate the minimum monthly rate needed to achieve the target IRR.
        
        Parameters:
        - duration: Loan duration in months
        - loan_amount: Principal loan amount in SAR
        - target_irr: Target annual IRR (decimal). If None, uses self.MIN_ANNUAL_IRR
        - repayment_structure: Type of repayment structure. Options:
            - 'monthly': Monthly equal payments (default)
            - 'bullet': Single payment at the end of the term
            - 'bi_monthly': Payment every 2 months
            - 'quarterly': Payment every 3 months
        
        Returns:
        - Minimum monthly rate required to achieve the target IRR
        """
        if target_irr is None:
            target_irr = self.MIN_ANNUAL_IRR
            
        # For simplicity and reliability, especially for the bullet payment structure,
        # we can use a straightforward calculation for monthly rate
        if repayment_structure == 'bullet':
            # Simplified formula for bullet payment
            # r = ((1+IRR)^(12/duration) - 1) / (1 - ADMIN_FEE_RATE)
            monthly_rate = ((1 + target_irr)**(1/12) - 1) * 1.2  # Adjust by 20% for admin fee effect
            return max(monthly_rate, self.MIN_VIABLE_RATE)
            
        # For monthly EMI, we can use a simpler approximation based on the target IRR
        if repayment_structure == 'monthly':
            # Simplified formula for monthly payment
            monthly_rate = target_irr / 12 * (1 + self.ADMIN_FEE_RATE/duration)
            return max(monthly_rate, self.MIN_VIABLE_RATE)
            
        # For other structures, use binary search as before but with more robust error handling
        min_rate = 0.0001  # Start with a very small rate
        max_rate = 0.10    # Cap at 10% monthly (very high)
        rate = target_irr / 12  # Start with target annual IRR divided by 12
        tolerance = 0.0001
        
        for _ in range(30):  # More iterations for better accuracy
            try:
                calculated_irr = self.calculate_irr(rate, duration, loan_amount, repayment_structure)
                
                if abs(calculated_irr - target_irr) < tolerance:
                    break
                    
                if calculated_irr < target_irr:
                    min_rate = rate
                    rate = (rate + max_rate) / 2
                else:
                    max_rate = rate
                    rate = (min_rate + rate) / 2
            except Exception as e:
                # If calculation fails, try a different rate
                rate = (min_rate + max_rate) / 2
        
        return max(rate, self.MIN_VIABLE_RATE)
    
    def get_rate(self, loan_amount, num_leads, cash_supply, duration, repayment_structure='monthly'):
        """
        Calculate the appropriate profit rate for a loan.
        
        Parameters:
        - loan_amount: Amount of the loan in SAR
        - num_leads: Number of leads in the pipeline (measure of demand)
        - cash_supply: Available cash to issue loans in SAR
        - duration: Instalment duration in months (1-6)
        - repayment_structure: Type of repayment structure. Options:
            - 'monthly': Monthly equal payments (default)
            - 'bullet': Single payment at the end of the term
            - 'bi_monthly': Payment every 2 months
            - 'quarterly': Payment every 3 months
        
        Returns:
        - Monthly profit rate as a decimal
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
        irr_based_rate = self.get_min_rate_for_irr(duration, loan_amount, None, repayment_structure)
        
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
    
    def calculate_loan_schedule(self, loan_amount, rate, duration, repayment_structure='monthly', include_admin_fee=True):
        """
        Calculate complete loan amortization schedule.
        
        Parameters:
        - loan_amount: Principal loan amount in SAR
        - rate: Monthly profit rate (decimal)
        - duration: Loan duration in months
        - repayment_structure: Type of repayment structure
        - include_admin_fee: Whether to include admin fee in calculation
        
        Returns:
        - DataFrame containing the loan schedule with monthly details
        """
        # Get cash flows to determine payment pattern
        cash_flows = self.get_cashflows(rate, duration, loan_amount, repayment_structure, include_admin_fee=False)
        cash_flows = cash_flows[1:]  # Remove initial outflow
        
        # Calculate admin fee
        admin_fee = loan_amount * self.ADMIN_FEE_RATE if include_admin_fee else 0
        net_disbursement = loan_amount - admin_fee if include_admin_fee else loan_amount
        
        # Create a monthly schedule dataframe
        schedule = pd.DataFrame(index=range(duration + 1))
        schedule['Month'] = range(duration + 1)
        schedule['Date'] = pd.date_range(start=pd.Timestamp.now().floor('D'), periods=duration + 1, freq='M')
        schedule['Payment'] = [0] + cash_flows
        schedule['Balance'] = 0.0
        schedule['Principal'] = 0.0
        schedule['Profit'] = 0.0
        schedule['Cumulative Principal'] = 0.0
        schedule['Cumulative Profit'] = 0.0
        
        # Set initial balance
        schedule.loc[0, 'Balance'] = loan_amount
        schedule.loc[0, 'Principal'] = -loan_amount
        schedule.loc[0, 'Profit'] = 0
        schedule.loc[0, 'Cumulative Principal'] = -loan_amount
        schedule.loc[0, 'Cumulative Profit'] = 0
        
        # For bullet payment, principal is paid at end, profit accrues monthly
        if repayment_structure == 'bullet':
            for i in range(1, duration + 1):
                schedule.loc[i, 'Balance'] = loan_amount
                schedule.loc[i, 'Profit'] = loan_amount * rate
                
                if i == duration:
                    # Final payment includes full principal
                    schedule.loc[i, 'Principal'] = loan_amount
                    schedule.loc[i, 'Balance'] = 0
                else:
                    schedule.loc[i, 'Principal'] = 0
            
        # For monthly payments
        elif repayment_structure == 'monthly':
            if rate > 0:
                monthly_payment = loan_amount * (rate * (1 + rate)**duration) / ((1 + rate)**duration - 1)
            else:
                monthly_payment = loan_amount / duration
                
            for i in range(1, duration + 1):
                # Get previous balance
                prev_balance = schedule.loc[i-1, 'Balance']
                
                # Calculate profit portion
                profit = prev_balance * rate
                
                # Calculate principal portion
                principal = monthly_payment - profit
                
                # Update balance
                new_balance = prev_balance - principal
                
                # For final payment, adjust to ensure ending balance is 0
                if i == duration:
                    principal = prev_balance
                    profit = monthly_payment - principal
                    new_balance = 0
                
                # Set values in dataframe
                schedule.loc[i, 'Balance'] = new_balance
                schedule.loc[i, 'Principal'] = principal
                schedule.loc[i, 'Profit'] = profit
        
        # For bi-monthly payments
        elif repayment_structure == 'bi_monthly':
            for i in range(1, duration + 1):
                # Get previous balance
                prev_balance = schedule.loc[i-1, 'Balance']
                
                # Calculate profit portion (accrues monthly)
                profit = prev_balance * rate
                
                # For payment months (every 2 months)
                if i % 2 == 0 or i == duration:
                    payment = schedule.loc[i, 'Payment']
                    
                    # Principal is payment minus accrued profit
                    if i % 2 == 0:  # Regular bi-monthly payment
                        accrued_profit = profit + schedule.loc[i-1, 'Profit']
                        principal = payment - accrued_profit
                    else:  # Final odd-month payment
                        principal = payment - profit
                    
                    # Update balance
                    new_balance = prev_balance - principal
                    
                    # Set values in dataframe
                    schedule.loc[i, 'Balance'] = new_balance
                    schedule.loc[i, 'Principal'] = principal
                    schedule.loc[i, 'Profit'] = profit if i == duration else accrued_profit
                else:
                    # Non-payment month - profit accrues
                    schedule.loc[i, 'Balance'] = prev_balance
                    schedule.loc[i, 'Principal'] = 0
                    schedule.loc[i, 'Profit'] = profit
        
        # For quarterly payments
        elif repayment_structure == 'quarterly':
            for i in range(1, duration + 1):
                # Get previous balance
                prev_balance = schedule.loc[i-1, 'Balance']
                
                # Calculate profit portion (accrues monthly)
                profit = prev_balance * rate
                
                # For payment months (every 3 months)
                if i % 3 == 0 or i == duration:
                    payment = schedule.loc[i, 'Payment']
                    
                    # Principal is payment minus accrued profit
                    if i % 3 == 0:  # Regular quarterly payment
                        accrued_profit = profit + schedule.loc[i-1, 'Profit'] + schedule.loc[i-2, 'Profit']
                        principal = payment - accrued_profit
                    elif i == duration and i % 3 == 1:  # Final payment 1 month after a quarterly payment
                        principal = payment - profit
                    else:  # Final payment 2 months after a quarterly payment
                        accrued_profit = profit + schedule.loc[i-1, 'Profit']
                        principal = payment - accrued_profit
                    
                    # Update balance
                    new_balance = prev_balance - principal
                    
                    # Set values in dataframe
                    schedule.loc[i, 'Balance'] = new_balance
                    schedule.loc[i, 'Principal'] = principal
                    schedule.loc[i, 'Profit'] = profit if i % 3 == 1 else (accrued_profit if i % 3 == 2 else profit + schedule.loc[i-1, 'Profit'] + schedule.loc[i-2, 'Profit'])
                else:
                    # Non-payment month - profit accrues
                    schedule.loc[i, 'Balance'] = prev_balance
                    schedule.loc[i, 'Principal'] = 0
                    schedule.loc[i, 'Profit'] = profit
        
        # Calculate cumulative values
        for i in range(1, duration + 1):
            schedule.loc[i, 'Cumulative Principal'] = schedule.loc[i-1, 'Cumulative Principal'] + schedule.loc[i, 'Principal']
            schedule.loc[i, 'Cumulative Profit'] = schedule.loc[i-1, 'Cumulative Profit'] + schedule.loc[i, 'Profit']
        
        # Add admin fee to the table
        if include_admin_fee:
            schedule.loc[0, 'Admin Fee'] = admin_fee
            schedule.loc[0, 'Net Disbursement'] = net_disbursement
        
        return schedule
    
    def compare_repayment_structures(self, loan_amount, rate, duration, include_admin_fee=True):
        """
        Compare IRRs and payment schedules for different repayment structures.
        
        Parameters:
        - loan_amount: Principal loan amount in SAR
        - rate: Monthly profit rate (decimal)
        - duration: Loan duration in months
        - include_admin_fee: Whether to include admin fee in calculation
        
        Returns:
        - DataFrame comparing key metrics across repayment structures
        """
        repayment_structures = list(self.repayment_structures.keys())
        results = []
        
        for structure in repayment_structures:
            # Get cash flows for this structure
            cash_flows = self.get_cashflows(rate, duration, loan_amount, structure, include_admin_fee)
            
            # Calculate IRR
            irr = self.calculate_irr(rate, duration, loan_amount, structure, include_admin_fee)
            
            # Calculate total payments and total profit
            total_payments = sum(cf for cf in cash_flows[1:] if cf > 0)
            total_profit = total_payments - loan_amount
            admin_fee = loan_amount * self.ADMIN_FEE_RATE if include_admin_fee else 0
            
            # Calculate number of payments
            num_payments = sum(1 for cf in cash_flows[1:] if cf > 0)
            
            # Create result entry
            structure_name = self.repayment_structures[structure]["display_name"]
            result = {
                "Repayment Structure": structure_name,
                "Annual IRR": irr,
                "Number of Payments": num_payments,
                "Total Payments": total_payments,
                "Total Profit": total_profit,
                "Admin Fee": admin_fee,
                "Total Cost": total_profit + admin_fee
            }
            results.append(result)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Format percentage columns
        comparison_df['Annual IRR'] = comparison_df['Annual IRR'].map(lambda x: f"{x:.2%}")
        
        return comparison_df
    
    def visualize_rate_matrix(self, rate_matrix, duration=None):
        """Create a heatmap visualization of the rate matrix."""
        plt.figure(figsize=(10, 8))
        
        # Format the values as percentages (using DataFrame.map instead of applymap)
        formatted_matrix = rate_matrix.copy()
        for col in formatted_matrix.columns:
            formatted_matrix[col] = formatted_matrix[col].map(lambda x: f"{x:.1%}")
        
        # Create a heatmap of the numeric values
        ax = sns.heatmap(rate_matrix, annot=formatted_matrix, fmt="", cmap="YlOrRd", 
                         linewidths=0.5, cbar_kws={'label': 'Monthly Profit Rate'})
        
        title = "BNPL Monthly Profit Rate Matrix"
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
    
    # Test with the specific example given
    test_loan_amount = 1_250_000
    test_rate = 0.01  # 1.00%
    test_duration = 6  # 6 months
    admin_fee_rate = 0.015  # 1.50%
    
    # Set the admin fee rate in calculator
    calculator.ADMIN_FEE_RATE = admin_fee_rate
    
    # Calculate IRR for each repayment structure
    print("\n===== TEST CASE =====")
    print(f"Loan Amount: {test_loan_amount:,} SAR")
    print(f"Monthly Rate: {test_rate:.2%}")
    print(f"Admin Fee Rate: {admin_fee_rate:.2%}")
    print(f"Duration: {test_duration} months")
    print("\nIRR for different repayment structures:")
    
    test_structures = ['monthly', 'bi_monthly', 'quarterly', 'bullet']
    for structure in test_structures:
        irr = calculator.calculate_irr(test_rate, test_duration, test_loan_amount, structure)
        display_name = calculator.repayment_structures[structure]['display_name']
        print(f"{display_name}: {irr:.2%}")
    
    # Calculate and display monthly EMI schedule
    print("\nMonthly EMI Schedule:")
    monthly_schedule = calculator.calculate_loan_schedule(test_loan_amount, test_rate, test_duration, 'monthly')
    print(monthly_schedule[['Month', 'Payment', 'Principal', 'Profit', 'Balance']].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    
    # Calculate and display bullet payment schedule
    print("\nBullet Payment Schedule:")
    bullet_schedule = calculator.calculate_loan_schedule(test_loan_amount, test_rate, test_duration, 'bullet')
    print(bullet_schedule[['Month', 'Payment', 'Principal', 'Profit', 'Balance']].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    
    # Additional examples
    print("\n===== ADDITIONAL EXAMPLES =====")
    print("Example: Rate for a 200,000 SAR loan with decent pipeline pressure (35 leads),")
    print("8M SAR cash supply, and 3-month duration:")
    rate = calculator.get_rate(200000, 35, 8_000_000, 3)
    print(f"Calculated rate: {rate:.2%}\n")
    
    # Compare different repayment structures
    print("\nComparing repayment structures for a 1,000,000 SAR loan with 1.00% monthly rate for 6 months:")
    comparison = calculator.compare_repayment_structures(1_000_000, 0.01, 6)
    print(comparison)
    
    # Calculate and print loan schedule for different repayment structures
    loan_amount = 1_000_000
    monthly_rate = 0.01
    duration = 6
    
    print("\nExample cashflows for different repayment structures:")
    for structure in calculator.repayment_structures.keys():
        print(f"\n{calculator.repayment_structures[structure]['display_name']} Cashflows:")
        cash_flows = calculator.get_cashflows(monthly_rate, duration, loan_amount, structure)
        print(f"Initial outflow: {cash_flows[0]:,.2f}")
        for i, cf in enumerate(cash_flows[1:], 1):
            if cf > 0:
                print(f"Month {i}: {cf:,.2f}")
    
    # Rate Matrix example
    print("\nRate Matrix for 150,000 SAR loan with 3-month duration:")
    rate_matrix = calculator.generate_rate_matrix(150000, 3)
    
    # Format and print the matrix
    formatted_matrix = rate_matrix.copy()
    for col in formatted_matrix.columns:
        formatted_matrix[col] = formatted_matrix[col].map(lambda x: f"{x:.2%}")
    print(formatted_matrix) 