import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from rate_calculator import BNPLRateCalculator
import matplotlib
matplotlib.use('Agg')

# Page configuration
st.set_page_config(
    page_title="BNPL Rate Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: 600;
    }
    .card {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0px;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 5px 10px;
        border-radius: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state to store calculator and current values
if 'calculator' not in st.session_state:
    st.session_state['calculator'] = BNPLRateCalculator()

# Helper functions
def format_as_percentage(value):
    """Format a decimal value as percentage."""
    return f"{value:.2%}"

def plot_heatmap(rate_matrix, duration):
    """Create a heatmap visualization using plotly."""
    # Convert rates to percentage strings for display
    percentage_matrix = rate_matrix.copy()
    for col in percentage_matrix.columns:
        percentage_matrix[col] = percentage_matrix[col].map(lambda x: f"{x:.2%}")
    
    # Create plotly figure
    fig = go.Figure(data=go.Heatmap(
        z=rate_matrix.values,
        x=rate_matrix.columns,
        y=rate_matrix.index,
        colorscale='YlOrRd',
        text=percentage_matrix.values,
        texttemplate="%{text}",
        textfont={"size":12},
    ))
    
    fig.update_layout(
        title=f"BNPL Monthly Interest Rate Matrix - {duration} Month Duration",
        xaxis_title="Cash Supply (SAR)",
        yaxis_title="Pipeline Pressure",
        height=500,
    )
    
    return fig

def run_monte_carlo_simulation(calculator, base_params, num_simulations=100):
    """
    Run a Monte Carlo simulation by varying parameters around their base values.
    
    Parameters:
    - calculator: BNPLRateCalculator instance
    - base_params: Dict with base parameter values
    - num_simulations: Number of simulations to run
    
    Returns:
    - DataFrame with simulation results
    """
    results = []
    
    for _ in range(num_simulations):
        # Vary parameters randomly within reasonable ranges
        loan_amount = base_params['loan_amount']
        
        # Vary the number of leads by ±30%
        num_leads_variation = np.random.uniform(0.7, 1.3)
        num_leads = max(1, int(base_params['num_leads'] * num_leads_variation))
        
        # Vary the cash supply by ±20%
        cash_variation = np.random.uniform(0.8, 1.2)
        cash_supply = max(calculator.min_cash, base_params['cash_supply'] * cash_variation)
        
        # Vary the duration by ±1 month (within bounds)
        duration_variation = np.random.randint(-1, 2)
        duration = max(calculator.min_duration, 
                       min(calculator.max_duration, 
                           base_params['duration'] + duration_variation))
        
        # Calculate the rate
        rate = calculator.get_rate(loan_amount, num_leads, cash_supply, duration)
        
        # Store the result
        results.append({
            'num_leads': num_leads,
            'cash_supply': cash_supply,
            'duration': duration,
            'rate': rate
        })
    
    return pd.DataFrame(results)

# Main interface
st.markdown('<div class="main-header">BNPL Rate Calculator</div>', unsafe_allow_html=True)
st.markdown(
    """
    This interactive dashboard allows you to calculate minimum acceptable monthly interest rates
    for B2B BNPL products based on pipeline pressure, cash supply, and instalment duration.
    """
)

# Create the main layout with tabs
tabs = st.tabs(["Rate Calculator", "Sensitivity Analysis", "Scenario Testing"])

with tabs[0]:
    st.markdown('<div class="sub-header">Rate Calculator</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Loan amount input
        loan_amount = st.number_input(
            "Loan Amount (SAR)",
            min_value=10000,
            max_value=10000000,
            value=200000,
            step=10000,
            help="Enter the loan amount in SAR. Loans up to 100k SAR have a fixed rate of 4%."
        )
        
        # Pipeline pressure (number of leads)
        num_leads = st.slider(
            "Pipeline Pressure (Number of Leads)",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            help="The number of leads in the pipeline, representing demand pressure."
        )
        
        # Cash supply
        cash_supply = st.slider(
            "Cash Supply (SAR)",
            min_value=int(st.session_state.calculator.min_cash),
            max_value=int(st.session_state.calculator.max_cash * 3),  # Allow for exceptional cash inflows
            value=8000000,
            step=1000000,
            format="%d SAR",
            help="Available cash to issue loans. Typically ranges from 2-12M SAR."
        )
        
        # Instalment duration
        duration = st.slider(
            "Instalment Duration (Months)",
            min_value=st.session_state.calculator.min_duration,
            max_value=st.session_state.calculator.max_duration,
            value=3,
            step=1,
            help="The duration of the instalment plan in months (1-6)."
        )
        
        # Calculate the rate
        rate = st.session_state.calculator.get_rate(loan_amount, num_leads, cash_supply, duration)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display the calculated rate
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Calculated Rate</div>', unsafe_allow_html=True)
        
        # Create a large, centered display for the rate
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #F3F4F6; border-radius: 5px;">
                <span style="font-size: 3rem; font-weight: bold; color: #1E3A8A;">{format_as_percentage(rate)}</span>
                <div style="font-size: 1.2rem; color: #4B5563;">Monthly Interest Rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display additional information based on the inputs
        pipeline_bucket = st.session_state.calculator.get_pipeline_bucket(num_leads)
        
        # Calculate the IRR based on the rate
        annual_irr = st.session_state.calculator.calculate_irr(rate, duration, loan_amount)
        
        # Calculate the minimum rate needed for the target IRR
        min_irr_rate = st.session_state.calculator.get_min_rate_for_irr(duration, loan_amount)
        
        # Calculate the negotiation range (between calculated rate and max rate)
        max_rate = st.session_state.calculator.MAX_RATE
        negotiation_range = max_rate - rate
        
        st.markdown(
            f"""
            <div class="info-text" style="margin-top: 15px;">
                <div><strong>Loan Size:</strong> {'Small (Fixed Rate)' if loan_amount <= st.session_state.calculator.SMALL_LOAN_THRESHOLD else 'Large (Dynamic Rate)'}</div>
                <div><strong>Pipeline Pressure:</strong> <span class="highlight">{pipeline_bucket.replace('_', ' ').title()}</span></div>
                <div><strong>Cash Supply Status:</strong> {('Low' if cash_supply < 6000000 else 'Medium' if cash_supply < 10000000 else 'High')}</div>
                <div><strong>Duration Impact:</strong> {'+' + format_as_percentage(rate - st.session_state.calculator.get_rate(loan_amount, num_leads, cash_supply, 1)) if duration > 1 else '0%'}</div>
                <div><strong>Annual IRR:</strong> <span class="highlight">{format_as_percentage(annual_irr)}</span></div>
                <div><strong>Min Rate for Target IRR:</strong> {format_as_percentage(min_irr_rate)}</div>
                <div><strong>Negotiation Range:</strong> {format_as_percentage(rate)} - {format_as_percentage(max_rate)} ({format_as_percentage(negotiation_range)} margin)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Generate and display the rate matrix
        rate_matrix = st.session_state.calculator.generate_rate_matrix(loan_amount, duration)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Rate Matrix</div>', unsafe_allow_html=True)
        
        fig = plot_heatmap(rate_matrix, duration)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            """
            <div class="info-text">
            The matrix shows minimum acceptable rates for different combinations of pipeline pressure (demand) 
            and cash supply levels. Higher pipeline pressure and lower cash supply result in higher rates.
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Duration impact visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Duration Impact Analysis</div>', unsafe_allow_html=True)
        
        # Generate rates for all durations
        durations = range(st.session_state.calculator.min_duration, st.session_state.calculator.max_duration + 1)
        duration_rates = [st.session_state.calculator.get_rate(loan_amount, num_leads, cash_supply, d) for d in durations]
        
        # Create a line chart
        duration_df = pd.DataFrame({
            'Duration (Months)': durations,
            'Rate': duration_rates
        })
        
        fig = px.line(
            duration_df, 
            x='Duration (Months)', 
            y='Rate',
            markers=True,
            labels={'Rate': 'Monthly Interest Rate'},
            title=f"Rate vs. Duration for {loan_amount:,} SAR Loan"
        )
        
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            """
            <div class="info-text">
            This chart shows how the monthly interest rate changes with different instalment durations,
            while keeping other parameters constant.
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="sub-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Explore how changes in input parameters affect the calculated interest rate and IRR.
        Use the sliders below to set the base values, then observe how varying each parameter
        impacts the rate and resulting IRR.
        """
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Base Parameters", unsafe_allow_html=False)
        
        # Base loan amount
        base_loan_amount = st.number_input(
            "Base Loan Amount (SAR)",
            min_value=10000,
            max_value=10000000,
            value=200000,
            step=10000,
            key="sensitivity_loan_amount"
        )
        
        # Base pipeline pressure
        base_num_leads = st.slider(
            "Base Pipeline Pressure (Leads)",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            key="sensitivity_leads"
        )
        
        # Base cash supply
        base_cash_supply = st.slider(
            "Base Cash Supply (SAR)",
            min_value=int(st.session_state.calculator.min_cash),
            max_value=int(st.session_state.calculator.max_cash * 2),
            value=8000000,
            step=1000000,
            format="%d SAR",
            key="sensitivity_cash"
        )
        
        # Base duration
        base_duration = st.slider(
            "Base Duration (Months)",
            min_value=st.session_state.calculator.min_duration,
            max_value=st.session_state.calculator.max_duration,
            value=3,
            step=1,
            key="sensitivity_duration"
        )
        
        # Calculate base rate
        base_rate = st.session_state.calculator.get_rate(
            base_loan_amount, base_num_leads, base_cash_supply, base_duration
        )
        
        st.markdown(f"**Base Rate:** {format_as_percentage(base_rate)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Create sensitivity analysis for pipeline pressure
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Pipeline Pressure Sensitivity", unsafe_allow_html=False)
        
        # Calculate rates for different lead counts
        lead_range = list(range(5, 101, 5))
        lead_rates = [st.session_state.calculator.get_rate(
            base_loan_amount, leads, base_cash_supply, base_duration
        ) for leads in lead_range]
        
        # Calculate IRRs for each rate
        lead_irrs = [st.session_state.calculator.calculate_irr(
            rate, base_duration, base_loan_amount
        ) for rate in lead_rates]
        
        lead_df = pd.DataFrame({
            'Pipeline Pressure (Leads)': lead_range,
            'Rate': lead_rates,
            'Annual IRR': lead_irrs
        })
        
        # Create a figure with secondary y-axis
        fig = go.Figure()
        
        # Add rate line
        fig.add_trace(
            go.Scatter(
                x=lead_df['Pipeline Pressure (Leads)'],
                y=lead_df['Rate'],
                mode='lines+markers',
                name='Monthly Rate',
                line=dict(color='blue')
            )
        )
        
        # Add IRR line
        fig.add_trace(
            go.Scatter(
                x=lead_df['Pipeline Pressure (Leads)'],
                y=lead_df['Annual IRR'],
                mode='lines+markers',
                name='Annual IRR',
                line=dict(color='green'),
                yaxis="y2"
            )
        )
        
        # Add a horizontal line at the target IRR
        fig.add_trace(
            go.Scatter(
                x=[lead_range[0], lead_range[-1]],
                y=[st.session_state.calculator.MIN_ANNUAL_IRR, st.session_state.calculator.MIN_ANNUAL_IRR],
                mode='lines',
                name='Target IRR',
                line=dict(color='red', dash='dash'),
                yaxis="y2"
            )
        )
        
        # Add a vertical line for the base value
        fig.add_vline(
            x=base_num_leads, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Base Value",
            annotation_position="top"
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title="Rate and IRR Sensitivity to Pipeline Pressure",
            xaxis_title="Pipeline Pressure (Leads)",
            yaxis=dict(
                title="Monthly Interest Rate",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                tickformat='.2%'
            ),
            yaxis2=dict(
                title="Annual IRR",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                anchor="x",
                overlaying="y",
                side="right",
                tickformat='.2%'
            ),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create sensitivity analysis for cash supply
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Cash Supply Sensitivity", unsafe_allow_html=False)
        
        # Calculate rates for different cash supply levels
        cash_range = list(range(
            int(st.session_state.calculator.min_cash),
            int(st.session_state.calculator.max_cash * 2) + 1,
            1000000
        ))
        cash_rates = [st.session_state.calculator.get_rate(
            base_loan_amount, base_num_leads, cash, base_duration
        ) for cash in cash_range]
        
        cash_df = pd.DataFrame({
            'Cash Supply (SAR)': cash_range,
            'Rate': cash_rates
        })
        
        fig = px.line(
            cash_df, 
            x='Cash Supply (SAR)', 
            y='Rate',
            markers=True,
            labels={'Rate': 'Monthly Interest Rate'},
            title="Rate Sensitivity to Cash Supply"
        )
        
        # Add a vertical line for the base value
        fig.add_vline(
            x=base_cash_supply, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Base Value",
            annotation_position="top"
        )
        
        fig.update_layout(
            yaxis_tickformat='.2%',
            xaxis_tickformat=',d'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="sub-header">Scenario Testing</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Test different scenarios and run Monte Carlo simulations to understand the variability
        in calculated rates under different conditions.
        """
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Scenario Parameters", unsafe_allow_html=False)
        
        # Scenario selection
        scenario = st.selectbox(
            "Select a Predefined Scenario",
            options=["Custom", "High Demand, Low Cash", "Low Demand, High Cash", "Balanced"],
            index=0
        )
        
        # Set parameters based on selected scenario
        if scenario == "High Demand, Low Cash":
            scenario_loan_amount = 250000
            scenario_num_leads = 75
            scenario_cash_supply = 3000000
            scenario_duration = 4
        elif scenario == "Low Demand, High Cash":
            scenario_loan_amount = 200000
            scenario_num_leads = 8
            scenario_cash_supply = 11000000
            scenario_duration = 2
        elif scenario == "Balanced":
            scenario_loan_amount = 180000
            scenario_num_leads = 30
            scenario_cash_supply = 7000000
            scenario_duration = 3
        else:  # Custom
            scenario_loan_amount = 200000
            scenario_num_leads = 35
            scenario_cash_supply = 8000000
            scenario_duration = 3
        
        # Inputs for scenario parameters
        scenario_loan_amount = st.number_input(
            "Loan Amount (SAR)",
            min_value=10000,
            max_value=10000000,
            value=scenario_loan_amount,
            step=10000,
            key="scenario_loan_amount"
        )
        
        scenario_num_leads = st.number_input(
            "Pipeline Pressure (Leads)",
            min_value=0,
            max_value=100,
            value=scenario_num_leads,
            step=5,
            key="scenario_num_leads"
        )
        
        scenario_cash_supply = st.number_input(
            "Cash Supply (SAR)",
            min_value=int(st.session_state.calculator.min_cash),
            max_value=int(st.session_state.calculator.max_cash * 3),
            value=scenario_cash_supply,
            step=1000000,
            format="%d",
            key="scenario_cash_supply"
        )
        
        scenario_duration = st.number_input(
            "Duration (Months)",
            min_value=st.session_state.calculator.min_duration,
            max_value=st.session_state.calculator.max_duration,
            value=scenario_duration,
            step=1,
            key="scenario_duration"
        )
        
        # Calculate scenario rate
        scenario_rate = st.session_state.calculator.get_rate(
            scenario_loan_amount, scenario_num_leads, scenario_cash_supply, scenario_duration
        )
        
        st.markdown(f"**Scenario Rate:** {format_as_percentage(scenario_rate)}")
        
        # Monte Carlo simulation parameters
        st.markdown("#### Monte Carlo Simulation", unsafe_allow_html=False)
        
        run_simulation = st.checkbox("Run Monte Carlo Simulation", value=True)
        
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Higher values give more accurate results but take longer to run."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if run_simulation:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Simulation Results", unsafe_allow_html=False)
            
            # Run the Monte Carlo simulation
            base_params = {
                'loan_amount': scenario_loan_amount,
                'num_leads': scenario_num_leads,
                'cash_supply': scenario_cash_supply,
                'duration': scenario_duration
            }
            
            simulation_results = run_monte_carlo_simulation(
                st.session_state.calculator, base_params, num_simulations
            )
            
            # Display summary statistics
            st.markdown("##### Rate Distribution Statistics")
            stats = simulation_results['rate'].describe()
            
            # Format statistics
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                'Value': [
                    f"{stats['count']:.0f}",
                    f"{stats['mean']:.2%}",
                    f"{stats['std']:.2%}",
                    f"{stats['min']:.2%}",
                    f"{stats['25%']:.2%}",
                    f"{stats['50%']:.2%}",
                    f"{stats['75%']:.2%}",
                    f"{stats['max']:.2%}"
                ]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(stats_df, hide_index=True)
            
            with col2:
                # Display confidence intervals
                confidence_intervals = [
                    ('90% CI', np.percentile(simulation_results['rate'], 5), np.percentile(simulation_results['rate'], 95)),
                    ('95% CI', np.percentile(simulation_results['rate'], 2.5), np.percentile(simulation_results['rate'], 97.5)),
                    ('99% CI', np.percentile(simulation_results['rate'], 0.5), np.percentile(simulation_results['rate'], 99.5))
                ]
                
                ci_df = pd.DataFrame({
                    'Confidence Interval': [ci[0] for ci in confidence_intervals],
                    'Lower Bound': [f"{ci[1]:.2%}" for ci in confidence_intervals],
                    'Upper Bound': [f"{ci[2]:.2%}" for ci in confidence_intervals]
                })
                
                st.dataframe(ci_df, hide_index=True)
            
            # Histogram of simulation results
            fig = px.histogram(
                simulation_results, 
                x='rate',
                nbins=20,
                labels={'rate': 'Monthly Interest Rate'},
                title=f"Distribution of Rates Across {num_simulations} Simulations"
            )
            
            # Add the base rate as a vertical line
            fig.add_vline(
                x=scenario_rate, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Base Rate",
                annotation_position="top"
            )
            
            fig.update_layout(
                xaxis_tickformat='.2%',
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot of relationships
            st.markdown("##### Relationship Between Parameters and Rates")
            
            param_to_plot = st.selectbox(
                "Select Parameter to Plot Against Rate:",
                options=["num_leads", "cash_supply", "duration"],
                format_func=lambda x: {
                    "num_leads": "Pipeline Pressure (Leads)",
                    "cash_supply": "Cash Supply (SAR)",
                    "duration": "Duration (Months)"
                }[x]
            )
            
            fig = px.scatter(
                simulation_results, 
                x=param_to_plot, 
                y='rate',
                labels={
                    'rate': 'Monthly Interest Rate',
                    'num_leads': 'Pipeline Pressure (Leads)',
                    'cash_supply': 'Cash Supply (SAR)',
                    'duration': 'Duration (Months)'
                },
                title=f"Rate vs. {param_to_plot.replace('_', ' ').title()}"
            )
            
            # Add a trend line
            fig.update_traces(marker=dict(size=8, opacity=0.6))
            fig.update_layout(
                yaxis_tickformat='.2%',
                height=450
            )
            
            if param_to_plot == "cash_supply":
                fig.update_layout(xaxis_tickformat=',d')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                ##### Monte Carlo Simulation
                
                Enable the Monte Carlo simulation on the left to see the distribution of possible rates
                when parameters are varied slightly from their base values.
                
                This helps understand the robustness of the rate calculation and identify
                how sensitive the rates are to small changes in inputs.
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

# Footer with explanation and documentation
st.markdown(
    """
    ---
    
    ### About the Rate Calculator
    
    This calculator determines minimum acceptable monthly interest rates for B2B BNPL products based on:
    
    - **Pipeline Pressure**: Measured by the number of leads, representing demand.
    - **Cash Supply**: Available capital to issue loans (typically 2-12M SAR).
    - **Instalment Duration**: Length of the loan term (1-6 months).
    
    #### Key Features:
    
    - **Fixed Rate**: 4% monthly rate for loans up to 100k SAR.
    - **Dynamic Rate**: For loans over 100k SAR, rates range from 1.8% to 3.5%.
    - **Economic Factors**: Rates are higher during high demand periods and when cash is limited.
    - **Duration Adjustment**: Longer instalment durations may result in higher rates.
    
    Use the sensitivity analysis and scenario testing to understand how different factors affect the calculated rates.
    """
) 