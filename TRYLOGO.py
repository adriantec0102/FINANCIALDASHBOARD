import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px

#Take the list of the S&P from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(url)
sp500_companies = sp500_table[0]
stocks = sp500_companies['Symbol'].tolist()  # List of stock symbols from S&P 500

# We can also filter out any unnecessary symbols
# for example, we can filter out any symbols with a '.' (like BRK.B)
stocks = [stock for stock in stocks if '.' not in stock]

# Function to get domain names(based on company name)
def fetch_company_domains():
    company_domains = {}
    for index, row in sp500_companies.iterrows():
        stock_symbol = row['Symbol']
        company_name = row['Security']
        company_name = company_name.lower().replace(" ", "")
        company_domains[stock_symbol] = company_name
    
    return company_domains

# Dictionary of stock symbols with their company domain names
company_domains = fetch_company_domains()

# Function to display company logo using Clearbit Logo API
def display_company_logo(stock_symbol):
    domain = company_domains.get(stock_symbol)
    if domain:
        logo_url = f"https://logo.clearbit.com/{domain}.com"
        response = requests.get(logo_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            st.image(img, width=200)
        else:
            st.write("Logo not available.")
    else:
        st.write("Logo not available.")

# Streamlit app setup
st.title('FINANCIAL DASHBOARD')
st.subheader('Done by Mario Perez for Financial Programming course')

pages = ['Stock Summary', 'Price Chart', 'Financials', 'Monte Carlo Simulation', 'News']
p_selection = st.sidebar.radio('Navigate to:', pages)

stock_symbol = st.sidebar.selectbox("Select a stock:", stocks)
update_button = st.sidebar.button("Update Data")

# UPDATE if pressed
if update_button:
    stock_data = yf.Ticker(stock_symbol)
else:
    stock_data = yf.Ticker(stocks[0])

chart_color = st.sidebar.color_picker("Pick a color for your charts", "#3667b0")

# Display company logo based on stock symbol
display_company_logo(stock_symbol)

# Page 1: Stock Summary
if p_selection == "Stock Summary":
    st.header("Stock Summary")
    stock_info = stock_data.info
    st.write(f"**Company:** {stock_info.get('longName', 'N/A')}")
    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
    st.write(f"**Description:** {stock_info.get('longBusinessSummary', 'N/A')}")

# Page 2: Price Chart
elif p_selection == "Price Chart":
    st.header("Price Chart")
    
    # Option to select the chart duration
    time_period = st.selectbox("Select Duration", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])
    time_mapping = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "3Y": "3y", "5Y": "5y", "MAX": "max"}
    historical_data = stock_data.history(period=time_mapping[time_period])
    
    if not historical_data.empty:
        # 50 day SMA
        historical_data['SMA50'] = historical_data['Close'].rolling(window=50).mean()

        # Select whether to show Candlestick or Line plot
        chart_type = st.selectbox("Select Chart Type", ["Candlestick", "Line"])
        
        if chart_type == "Candlestick":
            # Create candlestick chart using Plotly
            fig = go.Figure()

            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=historical_data.index,
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Close'],
                name="Candlesticks"
            ))

            # Add line for 50-day simple moving average
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['SMA50'],
                mode='lines',
                name='50-Day SMA',
                line=dict(color='orange')
            ))

            # Add volume trace
            fig.add_trace(go.Bar(
                x=historical_data.index,
                y=historical_data['Volume'],
                name='Volume',
                marker=dict(color='rgba(0, 0, 255, 0.2)')
            ))

            # Update layout to adjust chart
            fig.update_layout(
                title=f'{stock_symbol} Stock Price with Volume and 50-Day SMA',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )

        elif chart_type == "Line":
            # Create line chart using Plotly for closing price
            fig = go.Figure()

            # Add line for stock closing price
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue')
            ))

            # Add line for 50-day simple moving average
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['SMA50'],
                mode='lines',
                name='50-Day SMA',
                line=dict(color='orange')
            ))

            # Add volume trace
            fig.add_trace(go.Bar(
                x=historical_data.index,
                y=historical_data['Volume'],
                name='Volume',
                marker=dict(color='rgba(0, 0, 255, 0.2)')
            ))

            # Update layout for line chart
            fig.update_layout(
                title=f'{stock_symbol} Stock Price with Volume and 50-Day SMA (Line Chart)',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )

        st.plotly_chart(fig)
    else:
        st.write("No data available for this stock in the selected period.")

# Page 3: Financials
elif p_selection == "Financials":
    st.header("Financial Statements")
    statement_type = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period_type = st.radio("Period", ["Annual", "Quarterly"])

    if statement_type == "Income Statement":
        data = stock_data.financials if period_type == "Annual" else stock_data.quarterly_financials
    elif statement_type == "Balance Sheet":
        data = stock_data.balance_sheet if period_type == "Annual" else stock_data.quarterly_balance_sheet
    elif statement_type == "Cash Flow":
        data = stock_data.cashflow if period_type == "Annual" else stock_data.quarterly_cashflow

    st.write(data)

elif p_selection == "Monte Carlo Simulation":
    st.header("Monte Carlo Simulation")

    # Select the number of simulations and time horizon (e.g., 30, 60, or 90 days)
    num_simulations = st.selectbox("Select Number of Simulations", [200, 500, 1000])
    time_horizon = st.selectbox("Select Time Horizon (Days)", [30, 60, 90])

    # Fetch historical stock data
    historical_data = stock_data.history(period="1y")
    daily_returns = historical_data['Close'].pct_change()

    simulation_results = []

    for _ in range(num_simulations):
        # Simulate the daily returns for the given time horizon
        simulated_returns = np.random.choice(daily_returns[1:], size=time_horizon, replace=True)
        simulated_prices = [historical_data['Close'][-1]]  # Start from the last known price

        # Generate the simulated prices
        for return_value in simulated_returns:
            simulated_prices.append(simulated_prices[-1] * (1 + return_value))

        simulation_results.append(simulated_prices)

    # Convert the simulation results to a DataFrame for better handling
    simulation_results_df = pd.DataFrame(simulation_results)

    # Plot each simulation with a different color
    fig = plt.figure(figsize=(10, 6))
    for i in range(num_simulations):
        plt.plot(simulation_results_df.iloc[i, :], alpha=0.2)  # A little bit of transparency

    # Add the current stock price as a reference line
    current_price = historical_data['Close'][-1]
    plt.axhline(y=current_price, color='red', linestyle='--', label=f'Current Stock Price: {current_price:.2f}')

    # Calculate Value at Risk (VaR) at 95% confidence interval
    # We will calculate the 5th percentile (VaR at 95% confidence)
    var_95 = np.percentile(simulation_results_df.iloc[:, -1], 5)

    # Add the VaR to the chart
    plt.axhline(y=var_95, color='green', linestyle='--', label=f'VaR 95%: {var_95:.2f}')

    # Title and labels
    plt.title(f'Monte Carlo Simulation: {stock_symbol} Stock Price Predictions')
    plt.xlabel('Days')
    plt.ylabel('Price') 
    plt.legend()

    # Display the plot
    st.pyplot(fig)

    # Display VaR value
    st.write(f"Value at Risk (VaR) at 95% Confidence Interval: {var_95:.2f}")

# Page 5: Custom Analysis
elif p_selection == "News":
    st.header("Stock News")
    st.write("Stock news analysis")
    ticker = stock_symbol
    api_key = "f39c6ee8d5a54caabe18491a1edecde9" #We have to ask this API to the website for getting custom news per different stock symbol
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get("articles", [])
            if articles:
                for i, article in enumerate(articles[:10]):
                    st.subheader(f"News {i+1}")
                    st.write(f"**Published:** {article.get('publishedAt', 'N/A')}")
                    st.write(f"**Title:** {article.get('title', 'N/A')}")
                    st.write(f"**Summary:** {article.get('description', 'N/A')}")
            else:
                st.write("No news articles found for this stock.")
        else:
            st.error(f"Error fetching news: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
