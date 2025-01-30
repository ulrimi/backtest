#!/usr/bin/env python3
# main.py

import os
import csv
import time
import threading
import logging
import signal
import tempfile
import sys
from datetime import datetime, date, timedelta, time as dt_time

import pytz
import yfinance as yf
import pandas as pd
import streamlit as st
from decimal import Decimal, ROUND_HALF_UP
import uuid  # Ensure uuid is imported

# Schwab API imports (assuming you have a "schwab" package)
from schwab import auth, client
from schwab.client.base import BaseClient

# Local imports
from logger_setup import setup_logger


# ------------------------------------------------------------------------------------
# 1) Logger Setup
# ------------------------------------------------------------------------------------
# def initialize_logger(log_file):
#     """
#     Initialize the logger.
#     """
#     try:
#         logger = setup_logger(log_file)
#         logger.info("Logging system initialized.")
#         return logger
#     except Exception as e:
#         print(f"Failed to initialize logger: {e}")
#         sys.exit(1)

# logger = initialize_logger(LOG_FILE)
def initialize_logger():
    """
    Initialize the logger to stream to console.
    """
    try:
        logger = logging.getLogger("web_app_logger")
        logger.setLevel(logging.INFO)

        # Create a StreamHandler to output logs to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create a log formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)
        logger.propagate = False #prevent logs from being sent to root logger in app.py
        logger.info("Logging system initialized.")
        return logger
    except Exception as e:
        print(f"Failed to initialize logger: {e}")
        sys.exit(1)

logger = initialize_logger()

# ------------------------------------------------------------------------------------
# 2) Global Variables
# ------------------------------------------------------------------------------------


# For the Long Strategy
daily_high = None
stop_loss_long = None

# For the Short Strategy
daily_low = None
stop_loss_short = None

tx_log_df = []


# ------------------------------------------------------------------------------------
# 4) Data Fetching & Logging
# ------------------------------------------------------------------------------------

def reset_tx_log():
    """
    Reset the transaction log in session state to an empty DataFrame
    with the columns expected by your PnL function.
    """
    st.session_state.tx_log = pd.DataFrame(
        columns=[
            "Trade ID",
            "Timestamp",
            "Ticker",
            "Transaction Type",
            "Quantity",
            "Price",
            "Datetime"
        ]
    )

def log_transaction(quantity, price, transaction_type, candle_datetime, timestamp=None, trade_id=None):
    """
    Append a single trade row to st.session_state.tx_log (in-memory).
    Returns the trade_id (generated if not provided).
    """
    ticker = st.session_state.get("ticker", "UNKNOWN")# Default to "UNKNOWN" if not set
    if timestamp is None:
        timestamp = datetime.now()

    # If we need a Trade ID for a buy/short, generate one.
    if trade_id is None and transaction_type in ["buy", "short"]:
        trade_id = str(uuid.uuid4())  # Generate a unique ID
    elif trade_id is None:
        trade_id = "N/A"

    # Prepare a dictionary matching the columns in st.session_state.tx_log
    row_data = {
        "Trade ID":         trade_id,
        "Ticker":           ticker,                      # From your config
        "Transaction Type": transaction_type,
        "Quantity":         quantity,
        "Price":            round(price, 2),
        "Datetime":         candle_datetime.isoformat(),
    }

    # Convert the dict to a 1-row DataFrame with matching columns
    new_row_df = pd.DataFrame([row_data], columns=st.session_state.tx_log.columns)

    # Concatenate to the existing log
    st.session_state.tx_log = pd.concat(
        [st.session_state.tx_log, new_row_df],
        ignore_index=True
    )

    logger.info(
        f"Logged transaction: {transaction_type} {quantity} shares "
        f"at {price:.2f} on {timestamp}. Trade ID: {trade_id}"
    )
    return trade_id


def fetch_latest_candle(ticker):
    """Fetch 1-minute candle data for the last 30 days via yfinance."""
    try:
        ticker_data = yf.Ticker(ticker)
        all_data = []

        # Ensure we don't exceed the last 30 days
        max_days = 29
        today = datetime.now()
        earliest_date = today - timedelta(days=max_days)

        for i in range(6):
            # Calculate the end date (backwards in 5-day increments)
            end_date = today - timedelta(days=i * 5)
            start_date = end_date - timedelta(days=5)

            # Ensure start_date doesn't go past 30 days
            if start_date < earliest_date:
                start_date = earliest_date

            # Convert to yfinance's date format
            end_str = end_date.strftime('%Y-%m-%d')
            start_str = start_date.strftime('%Y-%m-%d')

            # Fetch data for this 5-day period
            df = ticker_data.history(start=start_str, end=end_str, interval="1m")

            # If data exists, process and append
            if not df.empty:
                df.reset_index(inplace=True)
                df.rename(columns={
                    "Datetime": "datetime",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }, inplace=True)
                all_data.append(df)

            # Break early if we've reached the earliest possible date
            if start_date <= earliest_date:
                break

        # Combine all dataframes
        if all_data:
            combined_df = pd.concat(all_data)
            combined_df.sort_values(by="datetime", inplace=True)  # Ensure data is in chronological order
            combined_df.drop_duplicates(subset="datetime", inplace=True)  # Remove any overlaps

            # Optional: print a sample
            logger.info(combined_df.head(3))


            return combined_df
        else:
            logger.warning("Received no data from yfinance.")
            return None

    except Exception as e:
        logger.error(f"Error fetching latest candle: {e}")
        return None


# def fetch_latest_candle_schwab(c, ticker,specific_date=None):
#     """
#     Fetches the latest candlestick data from Schwab's API, converts it to a DataFrame, and writes it to a CSV.
#     :param c: Schwab client instance.
#     :param ticker: Ticker symbol of the stock.
#     :return: The fetched DataFrame.
#     """
#     try:
#         # Define the timezone
#         utc = pytz.UTC
#         eastern = pytz.timezone('US/Eastern')

#         # Get the current date and time in Eastern Time
#         now_eastern = datetime.now(eastern)

#         specific_start_date = date(2025, 1, 13)
#         specific_end_date = date(2025, 1, 10) 

#         # Convert specific dates to datetime objects at start and end of the day in Eastern Time
#         start_of_day_eastern = eastern.localize(datetime.combine(specific_start_date, dt_time.min))
#         end_of_day_eastern = eastern.localize(datetime.combine(specific_end_date, dt_time.max))

        
#         # # Set the start of the day and end of the day in Eastern Time
#         # start_of_day_eastern = now_eastern.replace(hour=0, minute=0, second=0, microsecond=0)
#         # end_of_day_eastern = now_eastern.replace(hour=23, minute=59, second=59, microsecond=999999)

#         # Convert to UTC
#         start_of_day_utc = start_of_day_eastern.astimezone(utc)
#         end_of_day_utc = end_of_day_eastern.astimezone(utc)

#         # Format as ISO 8601 strings
#         start_datetime = start_of_day_utc
#         end_datetime = end_of_day_utc

#         logger.info(f"Start: {start_datetime}")
#         logger.info(f"End: {end_datetime}")

#         # Fetch candlestick data
#         response = c.get_price_history_every_minute(
#             ticker,
#             start_datetime=None,
#             end_datetime=None,
#             need_extended_hours_data=False,
#         )
#         data = response.json()  # Parse the JSON response

#         # Extract relevant data into a DataFrame
#         if "candles" in data:
#             df = pd.DataFrame(data["candles"])
#         else:
#             logger.info("No 'candles' key found in response.")
#             return None

#         if df.empty:
#             logger.info("No data fetched. DataFrame is empty.")
#             return None

#         # Convert datetime from milliseconds to human-readable format in Eastern Time
#         eastern = pytz.timezone('US/Eastern')
#         df['datetime'] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(eastern)

        # Define the CSV file path
        #output_dir = os.path.expanduser(f"~/Documents/pythonProjects/schwab-py/my-env/breakthroughDev/bbBackTestDev/{ticker}/csv")
        #os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        #output_file = os.path.join(output_dir, f"{ticker}_backtest.csv")

        # Write the DataFrame to a CSV file
        #df.to_csv(output_file, index=False)
        #logger.info(f"Data written to {output_file}")

        return df
    except Exception as e:
        logger.info(f"Error fetching or writing candle data: {e}")
        return None
    
# def check_auth():
#     # Initialize Supabase client
#     supabase_url = st.secrets["supabase"]["url"]
#     supabase_key = st.secrets["supabase"]["service_role"]
#     supabase = create_client(supabase_url, supabase_key)
    
#     if 'authenticated' not in st.session_state:
#         user = supabase.auth.get_user()
#         if user:
#             # Check token validity
#             token_data = supabase.table('user_tokens').select('*').eq('user_id', user.id).execute().data
#             if token_data:
#                 token_data = token_data[0]
#                 if datetime.now() < token_data['expires_at']:
#                     st.session_state.authenticated = True
#                     st.session_state.access_token = token_data['access_token']
#                 else:
#                     # Handle token refresh here
#                     pass
#             else:
#                 st.session_state.authenticated = False
#         else:
#             st.session_state.authenticated = False

# check_auth()

    
def run_backtest(df, params):
    reset_tx_log()

    try:
        # Split data into daily DataFrames
        daily_dfs = split_dataframe_by_day(df)
        daily_dfs = exclude_incomplete_days(daily_dfs)

        if not isinstance(daily_dfs, dict):
            raise ValueError("Data splitting failed.")

        # Process each day
        for date, daily_df in daily_dfs.items():
            apply_long_rules(daily_df, params)
            apply_short_rules(daily_df, params)

        # Calculate PnL
        return calculate_trade_pnl_with_percentage(st.session_state.tx_log)

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

def split_dataframe_by_day(df):
    """
    Splits a DataFrame into smaller DataFrames, one for each trading day.
    :param df: The original DataFrame with a datetime column in Eastern Time.
    :return: A dictionary where keys are trading days (YYYY-MM-DD) and values are the respective DataFrames.
    """
    try:
        # Ensure the datetime column is in the correct format
        if 'datetime' not in df.columns:
            logger.error("DataFrame does not contain a 'datetime' column.")
        df['date'] = df['datetime'].dt.date  # Extract the date part (YYYY-MM-DD)

        # Group by the date and split into separate DataFrames
        daily_dataframes = {date: group.drop(columns=['date']) for date, group in df.groupby('date')}

        return daily_dataframes
    except Exception as e:
        logger.info(f"Error splitting DataFrame by day: {e}")
        return None

# ------------------------------------------------------------------------------------
# 5) Trading Rule Functions
# ------------------------------------------------------------------------------------
def apply_long_rules(daily_df, params):
    logger.info(f"apply_long_rules received daily_df of type: {type(daily_df)}")
    if not isinstance(daily_df, pd.DataFrame):
        logger.error("daily_df is not a DataFrame.")
        return
    if daily_df.empty:
        logger.info("No data for this day.")
        return
    try:
        eod_stop_percentage = params['eod_stop_percentage']
        position_quantity = params['position_quantity']
        profit_target_multiplier = params['profit_target_multiplier']
        profit_target_sell_percentage = params['profit_target_sell_percentage']
        entry_multiplier = params['entry_multiplier']
        stop_loss_multiplier = params['stop_loss_multiplier']


        # Initialize variables
        daily_high = None
        stop_loss_long = None
        shares_held = 0  # Track the number of shares held
        entry_price = None  # Track entry price for profit target
        check_profit_target = False  # Flag to check profit target on next candle
        long_trades = []
        entry_target_set = False
        trade_closed = False
        trade_id = None
        long_trades_entered_today = 0

        # Pre-cutoff data to find initial daily_high
        pre_cutoff_data = daily_df[daily_df['datetime'].dt.time < dt_time(9, 55)]
        if not isinstance(pre_cutoff_data, pd.DataFrame):
            logger.error("pre_cutoff_data is not a DataFrame.")
            return
        if not pre_cutoff_data.empty:
            daily_high = pre_cutoff_data['high'].max()
            logger.info(f"Pre-cutoff daily high: {daily_high}")
        else:
            logger.info("No data available for pre-cutoff period.")
            return

        # Process post-cutoff data
        post_cutoff_data = daily_df[
            (daily_df['datetime'].dt.time >= dt_time(9, 55)) &
            (daily_df['datetime'].dt.time < dt_time(15, 50))
        ]

        for index, row in post_cutoff_data.iterrows():
            # Update daily high and stop loss
            if row['high'] > daily_high:
                daily_high = row['high']
                event_time = row['datetime']
                logger.info(f"New daily high detected: {daily_high} at {event_time}")
                if shares_held > 0:
                    stop_loss_long = round(daily_high * (1 - params['stop_loss_multiplier']), 2)
                    logger.info(f"Updated stop loss for active trade: {stop_loss_long}")
                elif not entry_target_set:
                    entry_target = round(daily_high * (1 + params['entry_multiplier']), 2)
                    stop_loss_long = round(daily_high * (1 - params['stop_loss_multiplier']), 2)
                    logger.info(f"Entry target set: {entry_target}, Stop loss set: {stop_loss_long}")
                    entry_target_set = True

            # Check for entry condition
            if entry_target_set and shares_held == 0 and long_trades_entered_today == 0:
                if row['low'] <= entry_target <= row['high']:
                    # Enter with POSITION_QUANTITY shares
                    if long_trades_entered_today < 1:
                        trade_id = log_transaction(
                            quantity= params['position_quantity'], 
                            price=entry_target, 
                            transaction_type="buy", 
                            candle_datetime=row['datetime'],
                            trade_id=trade_id
                        )
                        long_trades_entered_today = 1
                        logger.info(f'long_trades_entered_today {long_trades_entered_today}' )
                        shares_held = params['position_quantity']
                        entry_price = entry_target
                        check_profit_target = True
                        entry_price = entry_target
                        check_profit_target = True  # Check next candle for profit target
                        trade_entry_time = row['datetime']
                        trade_closed = False

            # Check profit target on the next candle after entry
            if check_profit_target:
                profit_target_price = round(entry_price * (1 + params['profit_target_multiplier']), 2)
                if row['low'] <= profit_target_price <= row['high']:
                    shares_to_sell = int(params['position_quantity'] * params['profit_target_sell_percentage']) 
                    log_transaction(
                        quantity=shares_to_sell,
                        price=profit_target_price,
                        transaction_type="sell",
                        candle_datetime=row['datetime'],
                        trade_id=trade_id
                    )
                    shares_held -= shares_to_sell
                    check_profit_target = False
                
            # Check stop loss for remaining shares
            if shares_held > 0 and not trade_closed:
                if row['low'] <= stop_loss_long:
                    # Sell all remaining shares
                    log_transaction(
                        quantity=shares_held, 
                        price=stop_loss_long, 
                        transaction_type="sell", 
                        candle_datetime=row['datetime'], 
                        trade_id=trade_id
                    )
                    logger.info(f"Stop loss hit: Sold {shares_held} shares at {stop_loss_long}")
                    shares_held = 0
                    trade_closed = True
                    pnl_percentage = ((stop_loss_long - entry_price) / entry_price) * 100
                    long_trades.append((trade_entry_time, row['datetime'], pnl_percentage))
                    break  # Exit loop after closing trade

        # EOD logic for remaining shares
        if shares_held > 0 and not trade_closed:
            eod_data = daily_df[
                (daily_df['datetime'].dt.time >= dt_time(15, 50)) &
                (daily_df['datetime'].dt.time < dt_time(16, 0))
            ]
            if not eod_data.empty:
                z = eod_stop_percentage / 6.0
                for eod_index, eod_row in eod_data.iterrows():
                    current_time = eod_row['datetime'].time()
                    # Update EOD stop loss each minute
                    if (15, 50) <= (current_time.hour, current_time.minute) < (15, 55):
                        multiplier = eod_stop_percentage
                    else:
                        minute_offset = max(0, current_time.minute - 54)
                        multiplier = eod_stop_percentage - z * minute_offset
                    eod_stop_loss = eod_row['low'] * (1 - multiplier)
                    eod_stop_loss = round(eod_stop_loss, 2)
                    if stop_loss_long is None or eod_stop_loss > stop_loss_long:
                        stop_loss_long = eod_stop_loss
                        logger.info(f"EOD updated stop loss to {stop_loss_long} at {eod_row['datetime']}")

                    # Check stop loss
                    if eod_row['low'] <= stop_loss_long:
                        log_transaction(
                            quantity=shares_held, 
                            price=stop_loss_long, 
                            transaction_type="sell", 
                            candle_datetime=row['datetime'],
                            trade_id=trade_id
                        )
                        logger.info(f"EOD stop loss hit: Sold {shares_held} shares at {stop_loss_long}")
                        shares_held = 0
                        trade_closed = True
                        break

                    # Close at 3:59 if still holding
                    if current_time.hour == 15 and current_time.minute == 59:
                        log_transaction(
                            quantity=shares_held, 
                            price=eod_row['close'], 
                            transaction_type="sell", 
                            candle_datetime=row['datetime'],
                            trade_id=trade_id
                        )
                        logger.info(f"Closed {shares_held} shares at EOD: {eod_row['close']}")
                        shares_held = 0
                        trade_closed = True
                        break
    except Exception as e:
        logger.error(f"Error in apply_long_rules(): {e}")

def apply_short_rules(daily_df, params):
    logger.info(f"apply_long_rules received daily_df of type: {type(daily_df)}")
    if not isinstance(daily_df, pd.DataFrame):
        logger.error("daily_df is not a DataFrame.")
        return
    if daily_df.empty:
        logger.info("No data for this day.")
        return
    try:
        logger.info("Applying short trading rules.")
        eod_stop_percentage = params['eod_stop_percentage']
        position_quantity = params['position_quantity']
        profit_target_multiplier = params['profit_target_multiplier']
        profit_target_sell_percentage = params['profit_target_sell_percentage']
        entry_multiplier = params['entry_multiplier']
        stop_loss_multiplier = params['stop_loss_multiplier']

        # Initialize variables
        daily_low = None
        stop_loss_short = None
        shares_held = 0  # Track the number of shares shorted
        entry_price = None
        check_profit_target = False
        short_trades = []
        entry_target_set = False
        trade_closed = False
        trade_id = None
        short_trades_entered_today = 0

        pre_cutoff_data = daily_df[daily_df['datetime'].dt.time < dt_time(9, 55)]
        if not isinstance(pre_cutoff_data, pd.DataFrame):
            logger.error("pre_cutoff_data is not a DataFrame.")
            return
        if not pre_cutoff_data.empty:
            daily_low = pre_cutoff_data['low'].min()
            logger.info(f"Pre-cutoff daily low: {daily_low}")
        else:
            logger.info("No data available for pre-cutoff period.")
            return

        post_cutoff_data = daily_df[
            (daily_df['datetime'].dt.time >= dt_time(9, 55)) &
            (daily_df['datetime'].dt.time < dt_time(15, 50))
        ]

        for index, row in post_cutoff_data.iterrows():
            if row['low'] < daily_low:
                daily_low = row['low']
                logger.info(f"New daily low detected: {daily_low}")
                if shares_held > 0:
                    stop_loss_short = round(daily_low * (1 + params['stop_loss_multiplier']), 2)
                    logger.info(f"Updated short stop loss: {stop_loss_short}")
                elif not entry_target_set:
                    entry_target = round(daily_low * (1 - params['entry_multiplier']), 2)
                    stop_loss_short = round(daily_low * (1 + params['stop_loss_multiplier']), 2)
                    logger.info(f"Short entry target: {entry_target}, Stop loss: {stop_loss_short}")
                    entry_target_set = True

            # Entry check
            if entry_target_set and shares_held == 0 and short_trades_entered_today == 0:
                if row['high'] >= entry_target >= row['low']:
                    if short_trades_entered_today < 1:
                        trade_id = log_transaction(
                            quantity= params['position_quantity'],  # Use config variable
                            price=entry_target, 
                            transaction_type="short", 
                            candle_datetime=row['datetime'],
                        )
                        short_trades_entered_today = 1
                        shares_held = params['position_quantity']  # Set to config variable
                        entry_price = entry_target
                        check_profit_target = True
                        logger.info(f"SHORT {params['position_quantity']} shares at {entry_target}")
                        entry_price = entry_target
                        check_profit_target = True
                        trade_entry_time = row['datetime']
                        trade_closed = False

            # Check profit target on next candle
            if check_profit_target:
                # Calculate profit target price for short (price goes DOWN)
                profit_target_price = round(entry_price * (1 - params['profit_target_multiplier']), 2) 
            
                if row['high'] >= profit_target_price >= row['low']:
                    # Calculate shares to cover based on percentage
                    shares_to_cover = int(params['position_quantity'] * params['profit_target_sell_percentage'])
                    
                    log_transaction(
                        quantity=shares_to_cover,
                        price=profit_target_price,
                        transaction_type="cover",
                        candle_datetime=row['datetime'],
                        trade_id=trade_id
                    )
                    shares_held -= shares_to_cover
                    logger.info(f"Partial COVER {shares_to_cover} shares at {profit_target_price}")
                    check_profit_target = False

            # Stop loss check
            if shares_held > 0 and not trade_closed:
                if row['high'] >= stop_loss_short:
                    log_transaction(
                        quantity=shares_held, 
                        price=stop_loss_short, 
                        transaction_type="cover", 
                        candle_datetime=row['datetime'], 
                        trade_id=trade_id
                    )
                    logger.info(f"Covered {shares_held} shares at stop loss: {stop_loss_short}")
                    shares_held = 0
                    trade_closed = True
                    pnl_percentage = ((entry_price - stop_loss_short) / entry_price) * 100
                    short_trades.append((trade_entry_time, row['datetime'], pnl_percentage))
                    break

        # EOD logic
        if shares_held > 0 and not trade_closed:
            eod_data = daily_df[
                (daily_df['datetime'].dt.time >= dt_time(15, 50)) &
                (daily_df['datetime'].dt.time < dt_time(16, 0))
            ]
            if not eod_data.empty:
                z = eod_stop_percentage / 6.0
                for eod_index, eod_row in eod_data.iterrows():
                    current_time = eod_row['datetime'].time()
                    if (15, 50) <= (current_time.hour, current_time.minute) < (15, 55):
                        multiplier = eod_stop_percentage
                    else:
                        minute_offset = max(0, current_time.minute - 54)
                        multiplier = eod_stop_percentage - z * minute_offset
                    eod_stop_loss = eod_row['high'] * (1 + multiplier)
                    eod_stop_loss = round(eod_stop_loss, 2)
                    if stop_loss_short is None or eod_stop_loss < stop_loss_short:
                        stop_loss_short = eod_stop_loss
                        logger.info(f"EOD updated short stop loss: {stop_loss_short}")

                    if eod_row['high'] >= stop_loss_short:
                        log_transaction(
                            quantity=shares_held, 
                            price=stop_loss_short, 
                            transaction_type="cover", 
                            candle_datetime=row['datetime'],
                            trade_id=trade_id
                        )
                        logger.info(f"EOD covered {shares_held} shares at {stop_loss_short}")
                        shares_held = 0
                        trade_closed = True
                        break

                    if current_time.hour == 15 and current_time.minute == 59:
                        log_transaction(
                            quantity=shares_held, 
                            price=eod_row['close'], 
                            transaction_type="cover", 
                            candle_datetime=row['datetime'], 
                            trade_id=trade_id
                        )
                        logger.info(f"EOD closed {shares_held} shares at {eod_row['close']}")
                        shares_held = 0
                        trade_closed = True
                        break
        # Log trades
        for entry_time, exit_time, pnl in short_trades:
            logger.info(f"Short trade: Entry {entry_time}, Exit {exit_time}, PnL%: {pnl:.2f}")
    except Exception as e:
        logger.error(f"Error in apply_short_rules(): {e}")

                    
# ------------------------------------------------------------------------------------
# 6) P&L Calculation Functions
# ------------------------------------------------------------------------------------
def calculate_trade_pnl_with_percentage(tx_log_df):
    """
    Calculate profit/loss, percentage gain/loss for each trade, overall return percentage,
    and separate PnL $ and % for long and short trades.
    """
    try:
        # Ensure the data is sorted by datetime
        tx_log_df['Datetime'] = pd.to_datetime(tx_log_df['Datetime'])
        df = tx_log_df.sort_values("Datetime").reset_index(drop=True)

        # Initialize variables
        trade_details = []  # List to hold all trade details
        total_pnl = 0.0
        total_return_pct = 0.0
        total_invested = 0.0
        trade_count = 0

        # Separate transactions by type
        buys   = df[df["Transaction Type"] == "buy"]
        sells  = df[df["Transaction Type"] == "sell"]
        shorts = df[df["Transaction Type"] == "short"]
        covers = df[df["Transaction Type"] == "cover"]

        trades_long  = len(buys)
        trades_short = len(shorts)

        # Initialize long and short metrics
        pnl_long   = 0.0
        invested_long = 0.0
        pnl_short  = 0.0
        invested_short = 0.0

        # Track used rows to prevent duplicates
        used_sells  = set()
        used_covers = set()

        #
        # 1) Process long trades (buy -> sell)
        #
        for _, buy_row in buys.iterrows():
            entry_price      = buy_row["Price"]
            quantity_bought  = buy_row["Quantity"]
            entry_datetime   = buy_row["Datetime"]
            trade_id         = buy_row["Trade ID"]

            # Track total invested
            total_invested   += entry_price * quantity_bought
            invested_long    += entry_price * quantity_bought

            # Find matching sells for this trade
            remaining_quantity = quantity_bought
            matching_sells     = sells[
                (sells["Trade ID"] == trade_id) & (~sells.index.isin(used_sells))
            ]

            # ----- Append the BUY row once -----
            trade_details.append({
                "Trade ID": trade_id,
                "Transaction Type": "buy",
                "Price": entry_price,
                "Quantity": quantity_bought,
                "Datetime": entry_datetime,
                "P&L": None,
                "Gain/Loss (%)": None,
            })

            # ----- Then loop over the sells -----
            for _, sell_row in matching_sells.iterrows():
                if remaining_quantity <= 0:
                    break

                sell_price   = sell_row["Price"]
                quantity_sold = min(sell_row["Quantity"], remaining_quantity)
                sell_datetime = sell_row["Datetime"]

                # Calculate P&L and gain/loss percentage
                pnl = (sell_price - entry_price) * quantity_sold
                gain_loss_pct = (
                    ((sell_price - entry_price) / entry_price) * 100
                    if entry_price != 0
                    else 0.0
                )

                trade_details.append({
                    "Trade ID": trade_id,
                    "Transaction Type": "sell",
                    "Price": sell_price,
                    "Quantity": quantity_sold,
                    "Datetime": sell_datetime,
                    "P&L": pnl,
                    "Gain/Loss (%)": gain_loss_pct,
                })

                # Update totals
                total_pnl       += pnl
                pnl_long        += pnl
                total_return_pct += gain_loss_pct
                remaining_quantity -= quantity_sold
                trade_count     += 1

                # Mark this sell as used
                used_sells.add(sell_row.name)

        #
        # 2) Process short trades (short -> cover)
        #
        for _, short_row in shorts.iterrows():
            entry_price       = short_row["Price"]
            quantity_shorted  = short_row["Quantity"]
            entry_datetime    = short_row["Datetime"]
            trade_id          = short_row["Trade ID"]

            # Track total invested
            total_invested    += entry_price * quantity_shorted
            invested_short    += entry_price * quantity_shorted

            # Find matching covers for this trade
            remaining_quantity = quantity_shorted
            matching_covers    = covers[
                (covers["Trade ID"] == trade_id) & (~covers.index.isin(used_covers))
            ]

            # ----- Append the SHORT row once -----
            trade_details.append({
                "Trade ID": trade_id,
                "Transaction Type": "short",
                "Price": entry_price,
                "Quantity": quantity_shorted,
                "Datetime": entry_datetime,
                "P&L": None,
                "Gain/Loss (%)": None,
            })

            # ----- Then loop over the covers -----
            for _, cover_row in matching_covers.iterrows():
                if remaining_quantity <= 0:
                    break

                cover_price    = cover_row["Price"]
                quantity_covered = min(cover_row["Quantity"], remaining_quantity)
                cover_datetime = cover_row["Datetime"]

                # Calculate P&L and gain/loss percentage
                pnl = (entry_price - cover_price) * quantity_covered
                gain_loss_pct = (
                    ((entry_price - cover_price) / entry_price) * 100
                    if entry_price != 0
                    else 0.0
                )

                trade_details.append({
                    "Trade ID": trade_id,
                    "Transaction Type": "cover",
                    "Price": cover_price,
                    "Quantity": quantity_covered,
                    "Datetime": cover_datetime,
                    "P&L": pnl,
                    "Gain/Loss (%)": gain_loss_pct,
                })

                # Update totals
                total_pnl       += pnl
                pnl_short       += pnl
                total_return_pct += gain_loss_pct
                remaining_quantity -= quantity_covered
                trade_count     += 1

                # Mark this cover as used
                used_covers.add(cover_row.name)

        # Convert to DataFrame
        trade_details_df = pd.DataFrame(trade_details)
        # Sort by datetime for proper ordering in the displayed DataFrame
        trade_details_df = trade_details_df.sort_values("Datetime").reset_index(drop=True)

        # Calculate average return percentage
        avg_return_pct = total_return_pct / trade_count if trade_count > 0 else 0.0

        # Calculate overall percentage return
        overall_return_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0.0

        # Calculate % returns for longs and shorts
        return_pct_long = (
            float(
                Decimal((pnl_long / invested_long) * 100)
                .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )
            if invested_long > 0
            else 0.0
        )

        return_pct_short = (
            float(
                Decimal((pnl_short / invested_short) * 100)
                .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )
            if invested_short > 0
            else 0.0
        )

        pnl_long  = round(pnl_long, 2)
        pnl_short = round(pnl_short, 2)

        return {
            "total_pnl": total_pnl,
            "avg_return_pct": avg_return_pct,
            "overall_return_pct": overall_return_pct,
            "trades": trade_details_df,
            "trades_long": trades_long,
            "trades_short": trades_short,
            "pnl_long": pnl_long,
            "return_pct_long": return_pct_long,
            "pnl_short": pnl_short,
            "return_pct_short": return_pct_short,
        }

    except Exception as e:
        logger.error(f"Error calculating P&L with percentages: {e}", exc_info=True)
        return {
            "total_pnl": 0.0,
            "avg_return_pct": 0.0,
            "overall_return_pct": 0.0,
            "trades": pd.DataFrame(),
            "pnl_long": 0.0,
            "return_pct_long": 0.0,
            "pnl_short": 0.0,
            "return_pct_short": 0.0,
        }

# ------------------------------------------------------------------------------------
# 7) Exclude Incomplete Days Function
# ------------------------------------------------------------------------------------
def exclude_incomplete_days(daily_dfs):
    """
    Exclude any day that is:
      1) The current day (if it's not yet past 4:00 PM).
      2) Missing a full 9:30 AM - 4:00 PM set of data.

    We'll check:
    - Earliest bar time <= 9:40 AM
    - Latest bar time >= 3:55 PM
    - Optionally, that the number of bars is ~ 390 (one bar per minute).
    """
    filtered = {}
    for date, df in daily_dfs.items():
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Skipping {date} as it's not a DataFrame.")
            continue
    
    # Define cutoff times
    earliest_cutoff = pd.Timestamp("09:40").time()
    latest_cutoff = pd.Timestamp("15:55").time()
    end_time = pd.Timestamp("16:00").time()
    
    # Current date in US/Eastern timezone
    eastern = pytz.timezone('US/Eastern')
    today = pd.Timestamp.now(tz=eastern).normalize()
    
    for date, df in daily_dfs.items():
        try:
            # Convert `date` to a Timestamp if necessary
            day_ts = pd.Timestamp(date).tz_localize(eastern)

            # If it's the current day and the market is still open or hasn't reached 4 PM yet, exclude it
            if day_ts == today:
                now_time = datetime.now(eastern).time()
                if now_time < end_time:
                    # It's today's trading day and not yet 4:00 PM -> skip
                    logger.info(f"Excluding current day {date} as trading is not yet complete.")
                    continue

            # If the day’s data is empty, skip
            if df.empty:
                logger.info(f"Excluding day {date} as it has no data.")
                continue

            # Check earliest and latest times in the DataFrame
            earliest_time = df["datetime"].dt.time.min()
            latest_time = df["datetime"].dt.time.max()

            # Condition 1: Earliest bar must be <= 9:40 AM
            # Condition 2: Latest bar must be >= 3:55 PM
            if earliest_time <= earliest_cutoff and latest_time >= latest_cutoff:
                # Optionally, check we have ~390 bars for the day:
                # Because actual data can vary if there’s a missing minute, 
                # you might do >= 389 or something slightly flexible.
                # Example check:
                total_bars = len(df)
                if total_bars >= 320:  # 390 bars is exact from 9:30–16:00 if on 1-min intervals
                    filtered[date] = df
                    logger.info(f"Including day {date} with {total_bars} bars.")
                else:
                    # Otherwise skip
                    logger.info(f"Excluding day {date} due to insufficient data bars ({total_bars} bars).")
                    continue
            else:
                # The day doesn't have a full 9:30–16:00 session
                logger.info(f"Excluding day {date} due to incomplete trading hours (Earliest: {earliest_time}, Latest: {latest_time}).")
                continue
        except Exception as e:
            logger.error(f"Error checking completeness for {date}: {e}")
            # Skip this day if we hit an error
            continue

    # Log how many days were excluded
    if len(filtered) != len(daily_dfs):
        logger.info(
            f"Excluded {len(daily_dfs) - len(filtered)} incomplete or current days. "
            f"Remaining days: {len(filtered)}."
        )
    return filtered

# ------------------------------------------------------------------------------------
# 8) Main Function
# ------------------------------------------------------------------------------------
def signal_handler(sig, frame):
    logger.info("Termination signal received. Exiting gracefully...")
    sys.exit(0)





