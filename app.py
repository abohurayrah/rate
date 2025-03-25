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
import json
matplotlib.use('Agg')

# Page configuration
st.set_page_config(
    page_title="BNPL Rate Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean color scheme
primary_color = "#0066cc"    # Primary blue
accent_color = "#4dabf7"     # Light blue
background_color = "#f8f9fa" # Light gray background
card_bg = "#ffffff"          # White for cards
text_color = "#333333"       # Dark gray for text
border_color = "#dee2e6"     # Light gray for borders
success_color = "#28a745"    # Green for success messages

# Custom CSS
st.markdown("""
<style>
    /* Base styles */
    .main-content {
        padding: 0 1rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.2rem;
        color: #0066cc;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        color: #0066cc;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #dee2e6;
    }
    
    /* Card styles */
    .card {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Results display */
    .result-display {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: 700;
        color: #0066cc;
    }
    
    .result-label {
        font-size: 1rem;
        color: #555;
        margin-top: 0.3rem;
    }
    
    /* Info panels */
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    
    .info-panel {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .info-panel-header {
        color: #555;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #f1f3f5;
        padding-bottom: 0.5rem;
    }
    
    .info-panel-content {
        display: grid;
        gap: 0.6rem;
    }
    
    .info-item {
        display: flex;
        justify-content: space-between;
    }
    
    .info-label {
        font-weight: 500;
    }
    
    .highlight-value {
        color: #0066cc;
        font-weight: 600;
    }
    
    /* Config panels */
    .config-panel {
        background-color: white;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .config-section {
        margin-bottom: 1.5rem;
    }
    
    .config-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0066cc;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e9ecef;
    }
    
    /* Other UI elements */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #0066cc !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
# Force reinitializing the calculator to pick up new method signatures
st.session_state['calculator'] = BNPLRateCalculator()

# Helper functions
def format_percentage(value):
    """Format decimal as percentage"""
    return f"{value:.2%}"

def format_currency(value):
    """Format number as currency"""
    return f"{value:,.0f} SAR"

def format_number(value):
    """Format number with comma separators"""
    return f"{value:,}"

# Create page header
st.markdown('<h1 class="main-header">BNPL Rate Calculator</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        Calculate minimum acceptable monthly profit rates for B2B BNPL products based on market conditions
    </div>
    """, 
    unsafe_allow_html=True
)

# Create tabs for main navigation
tabs = st.tabs(["Rate Calculator", "Repayment Schedule", "Sensitivity Analysis", "Configuration"])

# TAB 1: RATE CALCULATOR
with tabs[0]:
    st.markdown('<div class="section-header">Rate Calculator</div>', unsafe_allow_html=True)
    
    # Calculate rates and metrics
    rate = st.session_state.calculator.get_rate(200000, 35, 8000000, 3)  # Default values
    
    # Put matrix front and center
    matrix_col, control_col = st.columns([3, 2])
    
    with matrix_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Rate Matrix</div>', unsafe_allow_html=True)
        
        # Get input values from session state or use defaults
        loan_amount = st.session_state.get('loan_amount', 200000)
        num_leads = st.session_state.get('num_leads', 35)
        cash_supply = st.session_state.get('cash_supply', 8000000)
        duration = st.session_state.get('duration', 3)
        repayment_structure = st.session_state.get('repayment_structure', 'monthly')
        include_admin_fee = st.session_state.get('include_admin_fee', True)
        
        # Calculate updated rate and metrics based on inputs
        rate = st.session_state.calculator.get_rate(loan_amount, num_leads, cash_supply, duration, repayment_structure)
        annual_irr = st.session_state.calculator.calculate_irr(rate, duration, loan_amount, repayment_structure, include_admin_fee)
        min_irr_rate = st.session_state.calculator.get_min_rate_for_irr(duration, loan_amount, None, repayment_structure)
        max_rate = st.session_state.calculator.MAX_RATE
        negotiation_range = max_rate - rate
        
        # Calculate admin fee amount
        admin_fee = loan_amount * st.session_state.calculator.ADMIN_FEE_RATE if include_admin_fee else 0
        
        # Generate the rate matrix
        rate_matrix = st.session_state.calculator.generate_rate_matrix(loan_amount, duration)
        
        # Convert rates to percentage strings for display
        percentage_matrix = rate_matrix.copy()
        for col in percentage_matrix.columns:
            percentage_matrix[col] = percentage_matrix[col].map(lambda x: f"{x:.2%}")
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=rate_matrix.values,
            x=rate_matrix.columns,
            y=rate_matrix.index,
            colorscale='Blues',
            text=percentage_matrix.values,
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title=f"BNPL Monthly Profit Rate Matrix - {duration} Month Duration",
            xaxis_title="Cash Supply (SAR)",
            yaxis_title="Pipeline Pressure",
            height=500,  # Increased height
            margin=dict(l=60, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Result display
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                f"""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #0066cc;">{format_percentage(rate)}</div>
                    <div style="font-size: 0.9rem; color: #555; margin-top: 0.3rem;">Monthly Profit Rate</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col2:
            loan_size_class = "Small (Fixed Rate)" if loan_amount <= st.session_state.calculator.SMALL_LOAN_THRESHOLD else "Large (Dynamic Rate)"
            pipeline_bucket = st.session_state.calculator.get_pipeline_bucket(num_leads)
            st.markdown(
                f"""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
                    <div style="font-weight: 600; color: #0066cc; margin-bottom: 8px;">Key Metrics</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: 500;">Annual IRR:</span>
                        <span style="color: #0066cc; font-weight: 600;">{format_percentage(annual_irr)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: 500;">Min IRR Rate:</span>
                        <span>{format_percentage(min_irr_rate)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with control_col:
        # Input parameters card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Calculator Controls</div>', unsafe_allow_html=True)
        
        # Loan amount input
        loan_amount = st.number_input(
            "Loan Amount (SAR)",
            min_value=10000,
            max_value=10000000,
            value=loan_amount,
            step=10000,
            format="%d",
            help=f"Enter the loan amount in SAR. Loans up to {format_number(st.session_state.calculator.SMALL_LOAN_THRESHOLD)} SAR have a fixed rate."
        )
        st.caption(f"Current value: {format_number(loan_amount)} SAR")
        st.session_state['loan_amount'] = loan_amount
        
        # Pipeline pressure (number of leads)
        num_leads = st.slider(
            "Pipeline Pressure (Leads)",
            min_value=0,
            max_value=100,
            value=num_leads,
            step=5,
            help="Number of leads in the pipeline, representing demand pressure."
        )
        st.session_state['num_leads'] = num_leads
        
        # Cash supply with formatted values
        cash_min = int(st.session_state.calculator.min_cash)
        cash_max = int(st.session_state.calculator.max_cash * 3)
        
        cash_supply = st.slider(
            "Cash Supply (SAR)",
            min_value=cash_min,
            max_value=cash_max,
            value=cash_supply,
            step=1000000,
            format="%d",
            help=f"Available cash to issue loans. Typically ranges from {format_number(st.session_state.calculator.min_cash)} to {format_number(st.session_state.calculator.max_cash)} SAR."
        )
        st.caption(f"Current value: {format_number(cash_supply)} SAR")
        st.session_state['cash_supply'] = cash_supply
        
        # Instalment duration
        duration = st.slider(
            "Instalment Duration (Months)",
            min_value=st.session_state.calculator.min_duration,
            max_value=st.session_state.calculator.max_duration,
            value=duration,
            step=1,
            help="The duration of the instalment plan in months (1-6)."
        )
        st.session_state['duration'] = duration
        
        # Repayment structure selection
        repayment_structures = {k: v['display_name'] for k, v in st.session_state.calculator.repayment_structures.items()}
        repayment_structure = st.session_state.get('repayment_structure', 'monthly')
        
        selected_structure = st.selectbox(
            "Repayment Structure",
            options=list(repayment_structures.keys()),
            format_func=lambda x: repayment_structures[x],
            index=list(repayment_structures.keys()).index(repayment_structure),
            help="Choose how the loan will be repaid."
        )
        st.session_state['repayment_structure'] = selected_structure
        
        # Admin fee toggle
        include_admin_fee = st.session_state.get('include_admin_fee', True)
        include_admin_fee = st.checkbox(
            f"Include Admin Fee ({format_percentage(st.session_state.calculator.ADMIN_FEE_RATE)})",
            value=include_admin_fee,
            help="Whether to include admin fee in IRR calculations."
        )
        st.session_state['include_admin_fee'] = include_admin_fee
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Loan details card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Loan Details</div>', unsafe_allow_html=True)
        
        # Get contextual information
        cash_status = 'Low' if cash_supply < 6000000 else 'Medium' if cash_supply < 10000000 else 'High'
        duration_impact = rate - st.session_state.calculator.get_rate(loan_amount, num_leads, cash_supply, 1) if duration > 1 else 0
        
        # Calculate the cashflow sequence
        cash_flows = st.session_state.calculator.get_cashflows(rate, duration, loan_amount, repayment_structure, include_admin_fee)
        
        # Calculate total payments and profit
        total_payments = sum(cf for cf in cash_flows[1:] if cf > 0)
        total_profit = total_payments - loan_amount
        
        # Calculate number of payments
        num_payments = sum(1 for cf in cash_flows[1:] if cf > 0)
        
        # Using native Streamlit components to avoid raw HTML issues
        st.write("**Loan Size Classification:**", loan_size_class)
        st.write("**Pipeline Pressure:**", pipeline_bucket.replace('_', ' ').title())
        st.write("**Cash Supply:**", f"{format_number(cash_supply)} SAR ({cash_status})")
        st.write("**Duration Impact:**", f"+{format_percentage(duration_impact)}" if duration > 1 else "0%")
        
        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Repayment Details</div>', unsafe_allow_html=True)
        
        # Display repayment structure info
        structure_name = st.session_state.calculator.repayment_structures[repayment_structure]['display_name']
        st.write("**Repayment Structure:**", structure_name)
        st.write("**Number of Payments:**", f"{num_payments}")
        st.write("**Admin Fee:**", f"{format_number(admin_fee)} SAR")
        st.write("**Total Payments:**", f"{format_number(total_payments)} SAR")
        st.write("**Total Profit:**", f"{format_number(total_profit)} SAR")
        st.write("**Annual IRR:**", format_percentage(annual_irr))
        
        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Negotiation Range</div>', unsafe_allow_html=True)
        
        st.write("**Rate Range:**", f"{format_percentage(rate)} - {format_percentage(max_rate)}")
        st.write("**Rate Margin:**", format_percentage(negotiation_range))
        
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: REPAYMENT SCHEDULE
with tabs[1]:
    st.markdown('<div class="section-header">Repayment Schedule Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 1rem;">
            Analyze repayment schedules and cashflows for different repayment structures.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Get loan parameters from session state or use defaults
    loan_amount = st.session_state.get('loan_amount', 200000)
    rate = st.session_state.calculator.get_rate(
        loan_amount, 
        st.session_state.get('num_leads', 35),
        st.session_state.get('cash_supply', 8000000),
        st.session_state.get('duration', 3),
        st.session_state.get('repayment_structure', 'monthly')
    )
    duration = st.session_state.get('duration', 3)
    repayment_structure = st.session_state.get('repayment_structure', 'monthly')
    include_admin_fee = st.session_state.get('include_admin_fee', True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Loan Parameters</div>', unsafe_allow_html=True)
        
        # Allow user to customize parameters for this tab
        custom_loan_amount = st.number_input(
            "Loan Amount (SAR)",
            min_value=10000,
            max_value=10000000,
            value=loan_amount,
            step=10000,
            format="%d",
            key="schedule_loan_amount"
        )
        st.caption(f"Current value: {format_number(custom_loan_amount)} SAR")
        
        custom_rate = st.number_input(
            "Monthly Profit Rate",
            min_value=0.001,
            max_value=0.10,
            value=rate,
            step=0.001,
            format="%.3f",
            key="schedule_rate"
        )
        st.caption(f"Current value: {format_percentage(custom_rate)}")
        
        custom_duration = st.slider(
            "Loan Duration (Months)",
            min_value=1,
            max_value=12,
            value=duration,
            step=1,
            key="schedule_duration"
        )
        
        # Repayment structure selection for comparison
        structures = {k: v['display_name'] for k, v in st.session_state.calculator.repayment_structures.items()}
        custom_structure = st.selectbox(
            "Repayment Structure",
            options=list(structures.keys()),
            format_func=lambda x: structures[x],
            index=list(structures.keys()).index(repayment_structure),
            key="schedule_structure"
        )
        
        # Admin fee toggle
        custom_admin_fee = st.checkbox(
            f"Include Admin Fee ({format_percentage(st.session_state.calculator.ADMIN_FEE_RATE)})",
            value=include_admin_fee,
            key="schedule_admin_fee"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compare IRRs across structures
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Compare Repayment Structures</div>', unsafe_allow_html=True)
        
        comparison_df = st.session_state.calculator.compare_repayment_structures(
            custom_loan_amount, custom_rate, custom_duration, custom_admin_fee
        )
        
        # Format numeric columns
        display_df = comparison_df.copy()
        numeric_cols = ['Total Payments', 'Total Profit', 'Admin Fee', 'Total Cost']
        for col in numeric_cols:
            display_df[col] = display_df[col].map(lambda x: format_number(x))
            
        st.dataframe(display_df, use_container_width=True)
        
        # Generate a comparison chart
        fig = px.bar(
            comparison_df,
            x='Repayment Structure',
            y='Total Cost',
            title="Total Cost Comparison",
            color='Repayment Structure',
            text_auto=True
        )
        
        fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Amortization Schedule</div>', unsafe_allow_html=True)
        
        # Calculate loan schedule
        schedule = st.session_state.calculator.calculate_loan_schedule(
            custom_loan_amount, custom_rate, custom_duration, custom_structure, custom_admin_fee
        )
        
        # Select columns to display
        display_columns = ['Month', 'Date', 'Payment', 'Principal', 'Profit', 'Balance']
        if custom_admin_fee and 'Admin Fee' in schedule.columns:
            display_columns = ['Month', 'Date', 'Payment', 'Principal', 'Profit', 'Balance', 'Admin Fee']
        
        # Format numeric columns for display
        display_schedule = schedule[display_columns].copy()
        for col in ['Payment', 'Principal', 'Profit', 'Balance', 'Admin Fee']:
            if col in display_schedule.columns:
                display_schedule[col] = display_schedule[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        
        st.dataframe(display_schedule, use_container_width=True)
        
        # Generate a cashflow chart
        monthly_data = schedule[['Month', 'Payment', 'Principal', 'Profit']].copy()
        monthly_data.loc[0, 'Payment'] = -custom_loan_amount  # Show initial outflow correctly
        
        # Prepare data for chart
        chart_data = pd.melt(
            monthly_data,
            id_vars=['Month'],
            value_vars=['Payment'],
            var_name='Category',
            value_name='Amount'
        )
        
        # Create the chart
        fig = px.bar(
            chart_data,
            x='Month',
            y='Amount',
            color='Category',
            title="Loan Cashflow",
            labels={'Amount': 'SAR', 'Month': 'Month'}
        )
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash="solid", line_color="grey")
        
        # Update layout
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add balance line chart
        balance_fig = px.line(
            schedule,
            x='Month',
            y='Balance',
            title="Outstanding Balance",
            labels={'Balance': 'Outstanding Principal (SAR)', 'Month': 'Month'}
        )
        
        balance_fig.update_layout(height=300)
        st.plotly_chart(balance_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: SENSITIVITY ANALYSIS
with tabs[2]:
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 1rem;">
            Explore how changes in input parameters affect calculated rates and IRR.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Base parameter inputs
    param_col, chart_col = st.columns([1, 2])
    
    with param_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Base Parameters</div>', unsafe_allow_html=True)
        
        # Base loan amount with formatted values
        base_loan_amount = st.number_input(
            f"Base Loan Amount (SAR)",
            min_value=10000,
            max_value=10000000,
            value=200000,
            step=10000,
            format="%d",
            key="sensitivity_loan_amount"
        )
        st.write(f"Current value: {format_number(base_loan_amount)} SAR")
        
        # Base pipeline pressure
        base_num_leads = st.slider(
            "Base Pipeline Pressure (Leads)",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            key="sensitivity_leads"
        )
        
        # Base cash supply with formatted display
        base_cash_supply = st.slider(
            "Base Cash Supply (SAR)",
            min_value=int(st.session_state.calculator.min_cash),
            max_value=int(st.session_state.calculator.max_cash * 2),
            value=8000000,
            step=1000000,
            format="%d",
            key="sensitivity_cash"
        )
        st.write(f"Current value: {format_number(base_cash_supply)} SAR")
        
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
        base_irr = st.session_state.calculator.calculate_irr(
            base_rate, base_duration, base_loan_amount
        )
        
        # Display base values using native Streamlit components
        st.markdown("#### Base Rate & IRR", unsafe_allow_html=False)
        st.metric(label="Monthly Profit Rate", value=format_percentage(base_rate))
        st.metric(label="Annual IRR", value=format_percentage(base_irr))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col:
        # Sensitivity chart options
        sensitivity_param = st.radio(
            "Select parameter to analyze:",
            options=["Pipeline Pressure", "Cash Supply", "Duration"],
            horizontal=True
        )
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if sensitivity_param == "Pipeline Pressure":
            # Calculate rates for different lead counts
            lead_range = list(range(5, 101, 5))
            lead_rates = [st.session_state.calculator.get_rate(
                base_loan_amount, leads, base_cash_supply, base_duration, 'monthly'
            ) for leads in lead_range]
            
            # Calculate IRRs for each rate
            lead_irrs = [st.session_state.calculator.calculate_irr(
                rate, base_duration, base_loan_amount, 'monthly'
            ) for rate in lead_rates]
            
            # Create dataframe
            df = pd.DataFrame({
                'Pipeline Pressure (Leads)': lead_range,
                'Rate': lead_rates,
                'Annual IRR': lead_irrs
            })
            
            # Create chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=df['Pipeline Pressure (Leads)'],
                    y=df['Rate'],
                    mode='lines+markers',
                    name='Monthly Rate',
                    line=dict(color='#0066cc')
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Pipeline Pressure (Leads)'],
                    y=df['Annual IRR'],
                    mode='lines+markers',
                    name='Annual IRR',
                    line=dict(color='#28a745')
                ),
                secondary_y=True
            )
            
            # Add target IRR line
            fig.add_trace(
                go.Scatter(
                    x=[lead_range[0], lead_range[-1]],
                    y=[st.session_state.calculator.MIN_ANNUAL_IRR, st.session_state.calculator.MIN_ANNUAL_IRR],
                    mode='lines',
                    name='Target IRR',
                    line=dict(color='#dc3545', dash='dash')
                ),
                secondary_y=True
            )
            
            # Add vertical line at base value
            fig.add_vline(
                x=base_num_leads, 
                line_dash="dash", 
                line_color="#6c757d",
                annotation_text="Base Value",
                annotation_position="top"
            )
            
            # Update layout
            fig.update_layout(
                title="Impact of Pipeline Pressure on Rate and IRR",
                xaxis_title="Pipeline Pressure (Number of Leads)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            fig.update_yaxes(
                title_text="Monthly Profit Rate", 
                secondary_y=False, 
                tickformat='.2%'
            )
            
            fig.update_yaxes(
                title_text="Annual IRR", 
                secondary_y=True, 
                tickformat='.2%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif sensitivity_param == "Cash Supply":
            # Calculate rates for different cash supply levels
            cash_range = list(range(
                int(st.session_state.calculator.min_cash),
                int(st.session_state.calculator.max_cash * 2) + 1,
                1000000
            ))
            cash_rates = [st.session_state.calculator.get_rate(
                base_loan_amount, base_num_leads, cash, base_duration, 'monthly'
            ) for cash in cash_range]
            
            # Calculate IRRs for each rate
            cash_irrs = [st.session_state.calculator.calculate_irr(
                rate, base_duration, base_loan_amount, 'monthly'
            ) for rate in cash_rates]
            
            # Create dataframe
            df = pd.DataFrame({
                'Cash Supply (SAR)': cash_range,
                'Rate': cash_rates,
                'Annual IRR': cash_irrs
            })
            
            # Create chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=df['Cash Supply (SAR)'],
                    y=df['Rate'],
                    mode='lines+markers',
                    name='Monthly Rate',
                    line=dict(color='#0066cc')
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Cash Supply (SAR)'],
                    y=df['Annual IRR'],
                    mode='lines+markers',
                    name='Annual IRR',
                    line=dict(color='#28a745')
                ),
                secondary_y=True
            )
            
            # Add target IRR line
            fig.add_trace(
                go.Scatter(
                    x=[cash_range[0], cash_range[-1]],
                    y=[st.session_state.calculator.MIN_ANNUAL_IRR, st.session_state.calculator.MIN_ANNUAL_IRR],
                    mode='lines',
                    name='Target IRR',
                    line=dict(color='#dc3545', dash='dash')
                ),
                secondary_y=True
            )
            
            # Add vertical line at base value
            fig.add_vline(
                x=base_cash_supply, 
                line_dash="dash", 
                line_color="#6c757d",
                annotation_text="Base Value",
                annotation_position="top"
            )
            
            # Update layout
            fig.update_layout(
                title="Impact of Cash Supply on Rate and IRR",
                xaxis_title="Cash Supply (SAR)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=60, r=60, t=80, b=60),
                xaxis_tickformat=',d'
            )
            
            fig.update_yaxes(
                title_text="Monthly Profit Rate", 
                secondary_y=False, 
                tickformat='.2%'
            )
            
            fig.update_yaxes(
                title_text="Annual IRR", 
                secondary_y=True, 
                tickformat='.2%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Duration
            # Calculate rates for different durations
            duration_range = list(range(
                st.session_state.calculator.min_duration,
                st.session_state.calculator.max_duration + 1
            ))
            duration_rates = [st.session_state.calculator.get_rate(
                base_loan_amount, base_num_leads, base_cash_supply, duration, 'monthly'
            ) for duration in duration_range]
            
            # Calculate IRRs for each rate and duration
            duration_irrs = [st.session_state.calculator.calculate_irr(
                rate, duration, base_loan_amount, 'monthly'
            ) for rate, duration in zip(duration_rates, duration_range)]
            
            # Calculate minimum IRR rates
            min_irr_rates = [st.session_state.calculator.get_min_rate_for_irr(
                duration, base_loan_amount, None, 'monthly'
            ) for duration in duration_range]
            
            # Create dataframe
            df = pd.DataFrame({
                'Duration (Months)': duration_range,
                'Rate': duration_rates,
                'Annual IRR': duration_irrs,
                'Min IRR Rate': min_irr_rates
            })
            
            # Create chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add rate line
            fig.add_trace(
                go.Scatter(
                    x=df['Duration (Months)'],
                    y=df['Rate'],
                    mode='lines+markers',
                    name='Monthly Rate',
                    line=dict(color='#0066cc')
                ),
                secondary_y=False
            )
            
            # Add min IRR rate line
            fig.add_trace(
                go.Scatter(
                    x=df['Duration (Months)'],
                    y=df['Min IRR Rate'],
                    mode='lines+markers',
                    name='Min Rate for Target IRR',
                    line=dict(color='#fd7e14', dash='dot')
                ),
                secondary_y=False
            )
            
            # Add IRR line
            fig.add_trace(
                go.Scatter(
                    x=df['Duration (Months)'],
                    y=df['Annual IRR'],
                    mode='lines+markers',
                    name='Annual IRR',
                    line=dict(color='#28a745')
                ),
                secondary_y=True
            )
            
            # Add target IRR line
            fig.add_trace(
                go.Scatter(
                    x=[duration_range[0], duration_range[-1]],
                    y=[st.session_state.calculator.MIN_ANNUAL_IRR, st.session_state.calculator.MIN_ANNUAL_IRR],
                    mode='lines',
                    name='Target IRR',
                    line=dict(color='#dc3545', dash='dash')
                ),
                secondary_y=True
            )
            
            # Add vertical line at base value
            fig.add_vline(
                x=base_duration, 
                line_dash="dash", 
                line_color="#6c757d",
                annotation_text="Base Value",
                annotation_position="top"
            )
            
            # Update layout
            fig.update_layout(
                title="Impact of Duration on Rates and IRR",
                xaxis_title="Duration (Months)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            fig.update_yaxes(
                title_text="Monthly Profit Rate", 
                secondary_y=False, 
                tickformat='.2%'
            )
            
            fig.update_yaxes(
                title_text="Annual IRR", 
                secondary_y=True, 
                tickformat='.2%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration impact analysis
            if len(duration_rates) > 1:
                # Calculate rate increases
                rate_diffs = [duration_rates[i] - duration_rates[i-1] for i in range(1, len(duration_rates))]
                
                st.markdown("<div class='config-header'>Rate Compensation per Additional Month</div>", unsafe_allow_html=True)
                
                impact_data = {
                    "Month Transition": [f"{i} → {i+1}" for i in range(1, len(duration_rates))],
                    "Rate Increase": [format_percentage(diff) for diff in rate_diffs],
                    "% Increase": [f"{(diff/duration_rates[i-1]*100):.1f}%" for i, diff in enumerate(rate_diffs, 1)]
                }
                
                impact_df = pd.DataFrame(impact_data)
                st.table(impact_df)
                
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: CONFIGURATION
with tabs[3]:
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 1rem;">
            Configure the rate calculator parameters. All changes will affect the calculations immediately.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    try:
        # Load current configuration
        with open("config.json", "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error("Could not load configuration file. Using default values.")
        # Create default config
        config = st.session_state.calculator.config
    
    # Two-column layout for configuration
    config_left, config_right = st.columns(2)
    
    with config_left:
        # Economic Constants
        st.markdown('<div class="config-panel">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Economic Constants</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            cost_of_funds = st.number_input(
                "Cost of Funds",
                min_value=0.001,
                max_value=0.10,
                value=float(config.get("cost_of_funds", 0.016)),
                step=0.001,
                format="%.3f",
                help="Cost of funding loans (e.g., 0.016 = 1.6%)"
            )
            
            min_viable_rate = st.number_input(
                "Minimum Viable Rate",
                min_value=0.001,
                max_value=0.10,
                value=float(config.get("min_viable_rate", 0.018)),
                step=0.001,
                format="%.3f",
                help="Absolute minimum monthly profit rate"
            )
            
            max_rate = st.number_input(
                "Maximum Rate",
                min_value=0.001,
                max_value=0.10,
                value=float(config.get("max_rate", 0.032)),
                step=0.001,
                format="%.3f",
                help="Maximum monthly profit rate"
            )
        
        with col2:
            fixed_small_loan_rate = st.number_input(
                "Fixed Small Loan Rate",
                min_value=0.001,
                max_value=0.10,
                value=float(config.get("fixed_small_loan_rate", 0.04)),
                step=0.001,
                format="%.3f",
                help="Fixed rate for small loans"
            )
            
            small_loan_threshold = st.number_input(
                "Small Loan Threshold",
                min_value=10000,
                max_value=1000000,
                value=int(config.get("small_loan_threshold", 100000)),
                step=10000,
                format="%d",
                help="Threshold for small loans"
            )
            st.caption(f"Current: {format_number(small_loan_threshold)} SAR")
            
            min_annual_irr = st.number_input(
                "Target Annual IRR",
                min_value=0.05,
                max_value=0.50,
                value=float(config.get("min_annual_irr", 0.20)),
                step=0.01,
                format="%.2f",
                help="Minimum annual IRR target"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Rate Calculation Weights
        st.markdown('<div class="config-panel">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Rate Calculation Weights</div>', unsafe_allow_html=True)
        
        pipeline_pressure_weight = st.slider(
            "Pipeline Pressure Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(config.get("pipeline_pressure_weight", 0.6)),
            step=0.05,
            help="Weight for pipeline pressure in rate calculation"
        )
        
        cash_supply_weight = st.slider(
            "Cash Supply Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(config.get("cash_supply_weight", 0.4)),
            step=0.05,
            help="Weight for cash supply in rate calculation"
        )
        
        # Normalize weights to ensure they sum to 1
        total_weight = pipeline_pressure_weight + cash_supply_weight
        if total_weight != 0:
            normalized_pipeline_weight = pipeline_pressure_weight / total_weight
            normalized_cash_weight = cash_supply_weight / total_weight
            
            # Show normalized weights
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Pipeline Weight (Normalized)", value=f"{normalized_pipeline_weight:.2f}")
            with col2:
                st.metric(label="Cash Weight (Normalized)", value=f"{normalized_cash_weight:.2f}")
        
        max_duration_adjustment = st.slider(
            "Maximum Duration Adjustment",
            min_value=0.0,
            max_value=0.5,
            value=float(config.get("max_duration_adjustment", 0.15)),
            step=0.01,
            format="%.2f",
            help="Maximum increase for longest duration (e.g., 0.15 = 15%)"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with config_right:
        # Pipeline Pressure Settings
        st.markdown('<div class="config-panel">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Pipeline Pressure Settings</div>', unsafe_allow_html=True)
        
        # Pipeline buckets
        pipeline_buckets = config.get("pipeline_buckets", {
            "very_low": [0, 10],
            "low": [10, 25],
            "decent": [25, 50],
            "high": [50, 9999]
        })
        
        st.markdown("<strong>Pipeline Bucket Ranges</strong>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            very_low_max = st.number_input("Very Low Max", 
                value=pipeline_buckets["very_low"][1], min_value=1, max_value=100)
            
            low_max = st.number_input("Low Max", 
                value=pipeline_buckets["low"][1], min_value=very_low_max + 1, max_value=100)
            
            decent_max = st.number_input("Decent Max", 
                value=pipeline_buckets["decent"][1], min_value=low_max + 1, max_value=100)
        
        # Update pipeline buckets
        updated_pipeline_buckets = {
            "very_low": [0, very_low_max],
            "low": [very_low_max, low_max],
            "decent": [low_max, decent_max],
            "high": [decent_max, 9999]
        }
        
        # Pipeline weights
        pipeline_weights = config.get("pipeline_weights", {
            "very_low": 0.0,
            "low": 0.3,
            "decent": 0.7,
            "high": 1.0
        })
        
        st.markdown("<strong>Pipeline Pressure Factors</strong>", unsafe_allow_html=True)
        
        with col2:
            very_low_weight = st.number_input("Very Low Factor", 
                value=pipeline_weights["very_low"], min_value=0.0, max_value=1.0, step=0.1)
            
            low_weight = st.number_input("Low Factor", 
                value=pipeline_weights["low"], min_value=0.0, max_value=1.0, step=0.1)
            
            decent_weight = st.number_input("Decent Factor", 
                value=pipeline_weights["decent"], min_value=0.0, max_value=1.0, step=0.1)
            
            high_weight = st.number_input("High Factor", 
                value=pipeline_weights["high"], min_value=0.0, max_value=1.0, step=0.1)
        
        # Update pipeline weights
        updated_pipeline_weights = {
            "very_low": very_low_weight,
            "low": low_weight,
            "decent": decent_weight,
            "high": high_weight
        }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cash and Duration Settings
        st.markdown('<div class="config-panel">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">Cash & Duration Settings</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<strong>Cash Supply Settings</strong>", unsafe_allow_html=True)
            
            min_cash = st.number_input(
                "Min Cash Supply (SAR)",
                min_value=500000,
                max_value=10000000,
                value=int(config.get("min_cash", 2000000)),
                step=500000,
                format="%d"
            )
            st.caption(f"Current: {format_number(min_cash)} SAR")
            
            max_cash = st.number_input(
                "Max Cash Supply (SAR)",
                min_value=min_cash + 1000000,
                max_value=50000000,
                value=int(config.get("max_cash", 12000000)),
                step=1000000,
                format="%d"
            )
            st.caption(f"Current: {format_number(max_cash)} SAR")
            
            cash_buckets = st.number_input(
                "Cash Supply Buckets",
                min_value=2,
                max_value=10,
                value=int(config.get("cash_buckets", 5)),
                step=1
            )
        
        with col2:
            st.markdown("<strong>Duration Settings</strong>", unsafe_allow_html=True)
            
            min_duration = st.number_input(
                "Min Duration (Months)",
                min_value=1,
                max_value=12,
                value=int(config.get("min_duration", 1)),
                step=1
            )
            
            max_duration = st.number_input(
                "Max Duration (Months)",
                min_value=min_duration + 1,
                max_value=24,
                value=int(config.get("max_duration", 6)),
                step=1
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Save configuration button
    save_col1, save_col2 = st.columns([1, 3])
    
    with save_col1:
        if st.button("Save Configuration", use_container_width=True):
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
                
                with save_col2:
                    st.success("✅ Configuration saved successfully! Changes will apply immediately.")
            
            except Exception as e:
                with save_col2:
                    st.error(f"❌ Error saving configuration: {str(e)}")

# Footer
st.markdown("""
---

### About the Rate Calculator

This calculator determines minimum acceptable monthly profit rates for B2B BNPL products based on:

- **Pipeline Pressure**: Measured by the number of leads, representing demand.
- **Cash Supply**: Available capital to issue loans (typically 2-12M SAR).
- **Instalment Duration**: Length of the loan term (1-6 months).
- **Minimum IRR Requirements**: Ensures loan rates achieve target annual IRR of 20%.
- **Repayment Structure**: Supports different repayment schedules (monthly, bi-monthly, quarterly, or bullet).

#### Key Features:

- **Fixed Rate**: 4% monthly rate for loans up to 100k SAR.
- **Dynamic Rate**: For loans over 100k SAR, rates range from 1.8% to 3.2%.
- **Economic Factors**: Rates are higher during high demand periods and when cash is limited.
- **Duration Adjustment**: Longer instalment durations may result in higher rates.
- **IRR Optimization**: Rates are adjusted to ensure target annual IRR is achieved.
- **Multiple Repayment Structures**: Compare IRRs and payment schedules across different repayment options.
- **Admin Fee Support**: Include or exclude admin fees in IRR calculations.
- **Amortization Schedules**: View detailed payment schedules for any repayment structure.
""")