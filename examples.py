from rate_calculator import BNPLRateCalculator
import matplotlib.pyplot as plt

def basic_example():
    """Basic usage example with a single rate calculation."""
    calculator = BNPLRateCalculator()
    
    # Calculate rate for a specific scenario
    rate = calculator.get_rate(
        loan_amount=200000,  # 200k SAR
        num_leads=35,        # 35 leads (decent pressure)
        cash_supply=8000000, # 8M SAR cash available  
        duration=3           # 3-month instalment plan
    )
    
    # Calculate the IRR for this rate
    irr = calculator.calculate_irr(rate, 3, 200000)
    
    # Calculate minimum rate needed for target IRR
    min_irr_rate = calculator.get_min_rate_for_irr(3, 200000)
    
    print(f"Example: Rate for a 200,000 SAR loan with decent pipeline pressure (35 leads),")
    print(f"8M SAR cash supply, and 3-month duration:")
    print(f"Calculated rate: {rate:.2%}")
    print(f"Annual IRR: {irr:.2%}")
    print(f"Minimum rate needed for target IRR ({calculator.MIN_ANNUAL_IRR:.0%}): {min_irr_rate:.2%}")
    
    return rate, irr, min_irr_rate

def matrix_example():
    """Generate and display a rate matrix for a specific loan amount and duration."""
    calculator = BNPLRateCalculator()
    
    loan_amount = 150000  # 150k SAR
    duration = 3          # 3-month instalment plan
    
    rate_matrix = calculator.generate_rate_matrix(loan_amount, duration)
    
    print(f"\nRate Matrix for {loan_amount:,} SAR loan with {duration}-month duration:")
    
    # Format and print the matrix (using df.copy and map instead of applymap)
    formatted_matrix = rate_matrix.copy()
    for col in formatted_matrix.columns:
        formatted_matrix[col] = formatted_matrix[col].map(lambda x: f"{x:.2%}")
    print(formatted_matrix)
    
    # Visualize the matrix
    plt.figure(figsize=(10, 8))
    calculator.visualize_rate_matrix(rate_matrix, duration)
    plt.savefig(f"rate_matrix_{duration}month_{loan_amount}SAR.png")
    plt.close()
    
    return rate_matrix

def small_loan_example():
    """Example with a small loan (fixed rate)."""
    calculator = BNPLRateCalculator()
    
    loan_amount = 80000  # 80k SAR (below 100k threshold)
    
    # Get the fixed rate
    rate = calculator.get_rate(loan_amount, 30, 5000000, 4)
    
    print(f"\nRate for {loan_amount:,} SAR loan (below 100k threshold):")
    print(f"Fixed rate: {rate:.2%}")
    
    # Generate the matrix for small loans
    small_loan_matrix = calculator.generate_rate_matrix(loan_amount)
    
    print("\nRate Matrix for small loans (fixed rate regardless of conditions):")
    
    # Format and print the small loan matrix
    formatted_small_matrix = small_loan_matrix.copy()
    for col in formatted_small_matrix.columns:
        formatted_small_matrix[col] = formatted_small_matrix[col].map(lambda x: f"{x:.2%}")
    print(formatted_small_matrix)
    
    return rate, small_loan_matrix

def duration_comparison_example():
    """Compare rates across different durations."""
    calculator = BNPLRateCalculator()
    
    loan_amount = 200000  # 200k SAR
    num_leads = 30        # 30 leads
    cash_supply = 6000000 # 6M SAR
    
    print("\nDuration Comparison for a 200,000 SAR loan:")
    print("30 leads (decent pressure), 6M SAR cash supply")
    print("\nDuration | Rate    | Annual IRR | Min Rate for Target IRR | Rate Increase")
    print("-" * 75)
    
    rates = {}
    irrs = {}
    min_rates = {}
    prev_rate = None
    
    for duration in range(1, 7):
        rate = calculator.get_rate(loan_amount, num_leads, cash_supply, duration)
        irr = calculator.calculate_irr(rate, duration, loan_amount)
        min_rate = calculator.get_min_rate_for_irr(duration, loan_amount)
        
        rates[duration] = rate
        irrs[duration] = irr
        min_rates[duration] = min_rate
        
        if prev_rate is not None:
            rate_increase = rate - prev_rate
            rate_increase_pct = rate_increase / prev_rate * 100
            rate_increase_str = f"+{rate_increase:.2%} (+{rate_increase_pct:.1f}%)"
        else:
            rate_increase_str = "N/A"
            
        print(f"{duration} month  | {rate:.2%} | {irr:.2%}  | {min_rate:.2%}               | {rate_increase_str}")
        prev_rate = rate
    
    # Calculate the average rate increase per additional month
    total_increase = rates[6] - rates[1]
    avg_increase_per_month = total_increase / 5
    print(f"\nAverage rate increase per additional month: {avg_increase_per_month:.2%}")
    print(f"Average percent increase: {avg_increase_per_month / rates[1] * 100:.1f}% of base rate")
    
    return rates, irrs, min_rates

def exceptional_cash_example():
    """Example showing how the calculator handles exceptional cash inflows."""
    calculator = BNPLRateCalculator()
    
    loan_amount = 250000         # 250k SAR
    num_leads = 40               # 40 leads (decent pressure)
    normal_cash = 10000000       # 10M SAR (normal range)
    exceptional_cash = 30000000  # 30M SAR (exceptional inflow)
    duration = 4                 # 4-month instalment
    
    # Calculate rates
    normal_rate = calculator.get_rate(loan_amount, num_leads, normal_cash, duration)
    exceptional_rate = calculator.get_rate(loan_amount, num_leads, exceptional_cash, duration)
    
    print("\nExceptional Cash Inflow Example:")
    print(f"Loan: {loan_amount:,} SAR, Pipeline: 40 leads, Duration: {duration} months")
    print(f"Normal cash (10M SAR) rate: {normal_rate:.2%}")
    print(f"Exceptional cash (30M SAR) rate: {exceptional_rate:.2%}")
    print(f"Rate reduction: {(normal_rate - exceptional_rate) * 100:.2f} percentage points")
    
    return normal_rate, exceptional_rate

def generate_all_duration_matrices():
    """Generate and save matrices for all durations."""
    calculator = BNPLRateCalculator()
    loan_amount = 200000  # 200k SAR
    
    print("\nGenerating matrices for all durations...")
    
    # Get matrices for all durations
    matrices = calculator.generate_duration_matrices(loan_amount)
    
    # Visualize and save each matrix
    for duration, matrix in matrices.items():
        plt.figure(figsize=(10, 8))
        calculator.visualize_rate_matrix(matrix, duration)
        plt.savefig(f"rate_matrix_{duration}month.png")
        plt.close()
    
    print(f"Generated and saved matrices for {len(matrices)} durations.")
    
    return matrices

if __name__ == "__main__":
    print("BNPL Rate Calculator Examples")
    print("=" * 50)
    
    # Run all examples
    basic_example()
    matrix_example()
    small_loan_example()
    duration_comparison_example()
    exceptional_cash_example()
    generate_all_duration_matrices()
    
    print("\nAll examples completed. Check the generated PNG files for visualizations.") 