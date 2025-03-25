import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# Main interface
st.markdown('<div class="main-header">BNPL Rate Calculator</div>', unsafe_allow_html=True)
st.markdown(
    """
    This interactive dashboard allows you to calculate minimum acceptable monthly interest rates
    for B2B BNPL products based on pipeline pressure, cash supply, and instalment duration.
    """
)

# Create the main layout with tabs
tabs = st.tabs(["Rate Calculator", "Sensitivity Analysis", "Configuration"])

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
        
        # Calculate IRRs for each rate
        cash_irrs = [st.session_state.calculator.calculate_irr(
            rate, base_duration, base_loan_amount
        ) for rate in cash_rates]
        
        cash_df = pd.DataFrame({
            'Cash Supply (SAR)': cash_range,
            'Rate': cash_rates,
            'Annual IRR': cash_irrs
        })
        
        # Create a figure with secondary y-axis
        fig = go.Figure()
        
        # Add rate line
        fig.add_trace(
            go.Scatter(
                x=cash_df['Cash Supply (SAR)'],
                y=cash_df['Rate'],
                mode='lines+markers',
                name='Monthly Rate',
                line=dict(color='blue')
            )
        )
        
        # Add IRR line
        fig.add_trace(
            go.Scatter(
                x=cash_df['Cash Supply (SAR)'],
                y=cash_df['Annual IRR'],
                mode='lines+markers',
                name='Annual IRR',
                line=dict(color='green'),
                yaxis="y2"
            )
        )
        
        # Add a horizontal line at the target IRR
        fig.add_trace(
            go.Scatter(
                x=[cash_range[0], cash_range[-1]],
                y=[st.session_state.calculator.MIN_ANNUAL_IRR, st.session_state.calculator.MIN_ANNUAL_IRR],
                mode='lines',
                name='Target IRR',
                line=dict(color='red', dash='dash'),
                yaxis="y2"
            )
        )
        
        # Add a vertical line for the base value
        fig.add_vline(
            x=base_cash_supply, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Base Value",
            annotation_position="top"
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title="Rate and IRR Sensitivity to Cash Supply",
            xaxis_title="Cash Supply (SAR)",
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickformat=',d'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create duration impact analysis
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Duration Impact Analysis", unsafe_allow_html=False)
        
        # Calculate rates for different durations
        duration_range = list(range(
            st.session_state.calculator.min_duration,
            st.session_state.calculator.max_duration + 1
        ))
        duration_rates = [st.session_state.calculator.get_rate(
            base_loan_amount, base_num_leads, base_cash_supply, duration
        ) for duration in duration_range]
        
        # Calculate IRRs for each rate and duration
        duration_irrs = [st.session_state.calculator.calculate_irr(
            rate, duration, base_loan_amount
        ) for rate, duration in zip(duration_rates, duration_range)]
        
        # Calculate the minimum rate needed to achieve target IRR for each duration
        min_irr_rates = [st.session_state.calculator.get_min_rate_for_irr(
            duration, base_loan_amount
        ) for duration in duration_range]
        
        duration_df = pd.DataFrame({
            'Duration (Months)': duration_range,
            'Rate': duration_rates,
            'Annual IRR': duration_irrs,
            'Min IRR Rate': min_irr_rates
        })
        
        # Create a figure with secondary y-axis
        fig = go.Figure()
        
        # Add rate line
        fig.add_trace(
            go.Scatter(
                x=duration_df['Duration (Months)'],
                y=duration_df['Rate'],
                mode='lines+markers',
                name='Monthly Rate',
                line=dict(color='blue')
            )
        )
        
        # Add IRR line
        fig.add_trace(
            go.Scatter(
                x=duration_df['Duration (Months)'],
                y=duration_df['Annual IRR'],
                mode='lines+markers',
                name='Annual IRR',
                line=dict(color='green'),
                yaxis="y2"
            )
        )
        
        # Add minimum IRR rate line
        fig.add_trace(
            go.Scatter(
                x=duration_df['Duration (Months)'],
                y=duration_df['Min IRR Rate'],
                mode='lines+markers',
                name='Min Rate for Target IRR',
                line=dict(color='purple', dash='dot')
            )
        )
        
        # Add a horizontal line at the target IRR
        fig.add_trace(
            go.Scatter(
                x=[duration_range[0], duration_range[-1]],
                y=[st.session_state.calculator.MIN_ANNUAL_IRR, st.session_state.calculator.MIN_ANNUAL_IRR],
                mode='lines',
                name='Target IRR',
                line=dict(color='red', dash='dash'),
                yaxis="y2"
            )
        )
        
        # Add a vertical line for the base value
        fig.add_vline(
            x=base_duration, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Base Value",
            annotation_position="top"
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title="Impact of Duration on Rates and IRR",
            xaxis_title="Duration (Months)",
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
        
        # Show the compensation needed for each additional month
        st.markdown("#### Rate Compensation per Additional Month")
        
        # Calculate the rate differences between consecutive durations
        rate_diffs = [duration_rates[i] - duration_rates[i-1] for i in range(1, len(duration_rates))]
        
        # Create a table with the rate differences
        duration_diff_df = pd.DataFrame({
            'Additional Month': [f"{i} → {i+1}" for i in range(1, len(duration_rates))],
            'Rate Increase': rate_diffs,
            'Rate Increase (%)': [diff/duration_rates[i-1]*100 for i, diff in enumerate(rate_diffs, 1)]
        })
        
        # Format the values as percentages
        duration_diff_df['Rate Increase'] = duration_diff_df['Rate Increase'].map(lambda x: f"{x:.2%}")
        duration_diff_df['Rate Increase (%)'] = duration_diff_df['Rate Increase (%)'].map(lambda x: f"{x:.2f}%")
        
        st.table(duration_diff_df)
        
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        """
        View and edit the configuration parameters that control the rate calculator.
        Changes to these settings will affect how rates are calculated across the application.
        """
    )
    
    try:
        # Load current configuration
        with open("config.json", "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error("Could not load configuration file. Using default values.")
        # Create a default config if needed
        calculator = BNPLRateCalculator()
        config = calculator.config
    
    # Use two columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Economic Parameters", unsafe_allow_html=False)
        
        # Economic constants
        cost_of_funds = st.number_input(
            "Cost of Funds",
            min_value=0.001,
            max_value=0.10,
            value=float(config.get("cost_of_funds", 0.016)),
            step=0.001,
            format="%.3f",
            help="The cost of funding loans (e.g., 0.016 = 1.6%)"
        )
        
        min_viable_rate = st.number_input(
            "Minimum Viable Rate",
            min_value=0.001,
            max_value=0.10,
            value=float(config.get("min_viable_rate", 0.018)),
            step=0.001,
            format="%.3f",
            help="The absolute minimum monthly rate that can be charged"
        )
        
        max_rate = st.number_input(
            "Maximum Rate",
            min_value=0.001,
            max_value=0.10,
            value=float(config.get("max_rate", 0.032)),
            step=0.001,
            format="%.3f",
            help="The maximum monthly rate that can be charged"
        )
        
        fixed_small_loan_rate = st.number_input(
            "Fixed Small Loan Rate",
            min_value=0.001,
            max_value=0.10,
            value=float(config.get("fixed_small_loan_rate", 0.04)),
            step=0.001,
            format="%.3f",
            help="The fixed rate for small loans below the threshold"
        )
        
        small_loan_threshold = st.number_input(
            "Small Loan Threshold (SAR)",
            min_value=10000,
            max_value=1000000,
            value=int(config.get("small_loan_threshold", 100000)),
            step=10000,
            help="Loans below this amount will use the fixed small loan rate"
        )
        
        min_annual_irr = st.number_input(
            "Target Annual IRR",
            min_value=0.05,
            max_value=0.50,
            value=float(config.get("min_annual_irr", 0.20)),
            step=0.01,
            format="%.2f",
            help="The minimum annual IRR target for loans"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Rate Calculation Weights", unsafe_allow_html=False)
        
        pipeline_pressure_weight = st.slider(
            "Pipeline Pressure Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(config.get("pipeline_pressure_weight", 0.6)),
            step=0.05,
            help="Weight given to pipeline pressure in the rate calculation"
        )
        
        cash_supply_weight = st.slider(
            "Cash Supply Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(config.get("cash_supply_weight", 0.4)),
            step=0.05,
            help="Weight given to cash supply in the rate calculation"
        )
        
        max_duration_adjustment = st.slider(
            "Maximum Duration Adjustment",
            min_value=0.0,
            max_value=0.5,
            value=float(config.get("max_duration_adjustment", 0.15)),
            step=0.01,
            format="%.2f",
            help="Maximum increase factor for the longest duration (e.g., 0.15 = 15% increase)"
        )
        
        # Normalize weights to ensure they sum to 1
        total_weight = pipeline_pressure_weight + cash_supply_weight
        if total_weight != 0:
            normalized_pipeline_weight = pipeline_pressure_weight / total_weight
            normalized_cash_weight = cash_supply_weight / total_weight
            
            st.markdown(f"**Normalized Weights:** Pipeline: {normalized_pipeline_weight:.2f}, Cash: {normalized_cash_weight:.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Pipeline Pressure Settings", unsafe_allow_html=False)
        
        # Create input fields for pipeline buckets
        pipeline_buckets = config.get("pipeline_buckets", {
            "very_low": [0, 10],
            "low": [10, 25],
            "decent": [25, 50],
            "high": [50, 9999]
        })
        
        st.markdown("##### Pipeline Bucket Ranges (Number of Leads)")
        col1, col2 = st.columns(2)
        
        with col1:
            very_low_max = st.number_input("Very Low Max", value=pipeline_buckets["very_low"][1], min_value=1, max_value=100)
            low_max = st.number_input("Low Max", value=pipeline_buckets["low"][1], min_value=very_low_max + 1, max_value=100)
            decent_max = st.number_input("Decent Max", value=pipeline_buckets["decent"][1], min_value=low_max + 1, max_value=100)
        
        # Calculate pipeline buckets from inputs
        updated_pipeline_buckets = {
            "very_low": [0, very_low_max],
            "low": [very_low_max, low_max],
            "decent": [low_max, decent_max],
            "high": [decent_max, 9999]
        }
        
        # Create input fields for pipeline weights
        pipeline_weights = config.get("pipeline_weights", {
            "very_low": 0.0,
            "low": 0.3,
            "decent": 0.7,
            "high": 1.0
        })
        
        st.markdown("##### Pipeline Pressure Factors")
        with col2:
            very_low_weight = st.number_input("Very Low Factor", value=pipeline_weights["very_low"], min_value=0.0, max_value=1.0, step=0.1)
            low_weight = st.number_input("Low Factor", value=pipeline_weights["low"], min_value=0.0, max_value=1.0, step=0.1)
            decent_weight = st.number_input("Decent Factor", value=pipeline_weights["decent"], min_value=0.0, max_value=1.0, step=0.1)
            high_weight = st.number_input("High Factor", value=pipeline_weights["high"], min_value=0.0, max_value=1.0, step=0.1)
        
        updated_pipeline_weights = {
            "very_low": very_low_weight,
            "low": low_weight,
            "decent": decent_weight,
            "high": high_weight
        }
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Cash Supply Settings", unsafe_allow_html=False)
        
        min_cash = st.number_input(
            "Minimum Cash Supply (SAR)",
            min_value=500000,
            max_value=10000000,
            value=int(config.get("min_cash", 2000000)),
            step=500000,
            help="The minimum cash supply threshold used in rate calculations"
        )
        
        max_cash = st.number_input(
            "Maximum Cash Supply (SAR)",
            min_value=min_cash + 1000000,
            max_value=50000000,
            value=int(config.get("max_cash", 12000000)),
            step=1000000,
            help="The maximum cash supply threshold used in rate calculations"
        )
        
        cash_buckets = st.number_input(
            "Cash Supply Buckets",
            min_value=2,
            max_value=10,
            value=int(config.get("cash_buckets", 5)),
            step=1,
            help="Number of divisions for cash supply ranges"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Duration Settings", unsafe_allow_html=False)
        
        min_duration = st.number_input(
            "Minimum Duration (Months)",
            min_value=1,
            max_value=12,
            value=int(config.get("min_duration", 1)),
            step=1,
            help="The minimum loan duration in months"
        )
        
        max_duration = st.number_input(
            "Maximum Duration (Months)",
            min_value=min_duration + 1,
            max_value=24,
            value=int(config.get("max_duration", 6)),
            step=1,
            help="The maximum loan duration in months"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Save configuration button
    if st.button("Save Configuration"):
        try:
            # Prepare updated configuration
            updated_config = {
                "cost_of_funds": cost_of_funds,
                "min_viable_rate": min_viable_rate,
                "max_rate": max_rate,
                "fixed_small_loan_rate": fixed_small_loan_rate,
                "small_loan_threshold": small_loan_threshold,
                "min_annual_irr": min_annual_irr,
                
                "pipeline_buckets": updated_pipeline_buckets,
                "pipeline_weights": updated_pipeline_weights,
                
                "min_cash": min_cash,
                "max_cash": max_cash,
                "cash_buckets": cash_buckets,
                
                "min_duration": min_duration,
                "max_duration": max_duration,
                
                "pipeline_pressure_weight": normalized_pipeline_weight if total_weight != 0 else 0.6,
                "cash_supply_weight": normalized_cash_weight if total_weight != 0 else 0.4,
                "max_duration_adjustment": max_duration_adjustment
            }
            
            # Save to file
            with open("config.json", "w") as f:
                json.dump(updated_config, f, indent=4)
            
            # Update calculator instance
            st.session_state.calculator = BNPLRateCalculator("config.json")
            
            st.success("Configuration saved successfully! Changes will apply to all calculations.")
            
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")

# Footer with explanation and documentation
st.markdown(
    """
    ---
    
    ### About the Rate Calculator
    
    This calculator determines minimum acceptable monthly interest rates for B2B BNPL products based on:
    
    - **Pipeline Pressure**: Measured by the number of leads, representing demand.
    - **Cash Supply**: Available capital to issue loans (typically 2-12M SAR).
    - **Instalment Duration**: Length of the loan term (1-6 months).
    - **Minimum IRR Requirements**: Ensures loan rates achieve target annual IRR of 20%.
    
    #### Key Features:
    
    - **Fixed Rate**: 4% monthly rate for loans up to 100k SAR.
    - **Dynamic Rate**: For loans over 100k SAR, rates range from 1.8% to 3.2%.
    - **Economic Factors**: Rates are higher during high demand periods and when cash is limited.
    - **Duration Adjustment**: Longer instalment durations may result in higher rates.
    - **IRR Optimization**: Rates are adjusted to ensure target annual IRR is achieved.
    
    Use the sensitivity analysis and scenario testing to understand how different factors affect the calculated rates and IRR.
    """
) 