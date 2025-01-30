#app.py

import sys
import logging
import pandas as pd
import streamlit as st
from datetime import datetime
from main import (
    fetch_latest_candle,
    #fetch_latest_candle_schwab,
    run_backtest,
    calculate_trade_pnl_with_percentage,
    reset_tx_log
)
import bcrypt


def check_authentication():
    # Initialize authentication status
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        # supabase_url = st.secrets["supabase"]["url"]
        # supabase_key = st.secrets["supabase"]["service_role"]
        # supabase = create_client(supabase_url, supabase_key)

        # user = supabase.auth.get_user()
        # if user:
        #     # Check token validity
        #     token_data = supabase.table('user_tokens').select('*').eq('user_id', user.id).execute().data[0]
        #     if datetime.now() < token_data['expires_at']:
        #         st.session_state.authenticated = True
        #         st.session_state.access_token = token_data['access_token']
        #     else:
        #         # Handle token refresh here
        #         pass
        # else:
        #     st.session_state.authenticated = False

    # Show password form if not authenticated
    if not st.session_state.authenticated:
        st.title("🔒 Backtester Access")
        
        with st.form("password_form"):
            password = st.text_input("Enter access password:", 
                                   type="password",
                                   help="Contact support for access credentials")
            submitted = st.form_submit_button("Unlock")
            
            if submitted:
                # Get stored hash from secrets (see security note below)
                correct_password_hash = st.secrets["access"]["password_hash"].encode()
                
                # Verify password against hash
                if bcrypt.checkpw(password.encode(), correct_password_hash):
                    st.session_state.authenticated = True

                else:
                    st.error("Incorrect password. Please try again.")
        
        st.stop()  # Stop execution here if not authenticated

check_authentication() # Enforce auth before anything else runs
# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session():
    if 'tx_log' not in st.session_state:
        st.session_state.tx_log = pd.DataFrame(columns=[
            'timestamp', 'trade_type', 'entry_price', 'exit_price', 
            'quantity', 'pnl', 'return_pct'
        ])
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()



def main():
    initialize_session()
    
    st.title("Breakout/Breakdown Backtester")
    st.sidebar.info(
    "⚠️ **Important**: After editing any parameter, you must click **Save Parameters** "
    "to ensure the changes are applied to the next backtest.")
    # Client initialization section
    with st.sidebar.form("params_form"):
        # Trading hours    
        # Strategy parameters
        st.text_input("Ticker", value="MSTR", key="ticker")
        st.number_input("Stop Loss Multiplier .01 = 1%", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", key="stop_loss_multiplier")
        st.number_input("Entry Multiplier  .01 = 1%", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", key="entry_multiplier")
        st.number_input("EOD Stop  .01 = 1%", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", key="eod_stop_percentage")
        st.number_input("Profit Target Multiplier  .01 = 1%", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", key="profit_target_multiplier")
        st.number_input("Position Quantity", min_value=1, value=10, key="position_quantity")
        st.number_input("PT Shares to Sell (% of position) .1 = 10%", min_value=0.0001, max_value=1.0, value=0.5, step=0.1, format="%.1f", key="profit_target_sell_percentage")

        # Add submit button with confirmation
        submitted = st.form_submit_button("Save Parameters")
        if submitted:
            st.success("Parameters saved successfully!")
            st.toast("Parameters updated successfully! 🎉")

    # Data loading section
    with st.expander("Load Historical Data.", expanded=True):
        symbol = st.session_state.ticker
        st.info('This will fetch the last 29 calendar (not trading) days of 1m candles from yfinance. You can fetch once, and run as many backtests as you want against that ticker.  You only need to Load Data again if you change the Ticker')
        if st.button("Load Data"):
 
            try:
                with st.spinner("Fetching historical data..."):
                    data = fetch_latest_candle(
                        symbol
                    )
                    if not data.empty:
                        st.session_state.historical_data = data
                        st.session_state.data_loaded = True
                        st.success(f"Loaded {len(data)} candles for {symbol}!")
                    else:
                        st.warning("No data returned from API")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Backtest configuration
    with st.expander("After loading Historical Data...", expanded=True):
        st.info("This will run the last saved parameters against the loaded historical data.")
        if st.button("Run Backtest"):
            if not st.session_state.data_loaded:
                st.warning("Load historical data first!")
                return
                
            try:
                backtest_params = {
                    'eod_stop_percentage': st.session_state.eod_stop_percentage,
                    'position_quantity': st.session_state.position_quantity,
                    'profit_target_multiplier': st.session_state.profit_target_multiplier,
                    'profit_target_sell_percentage': st.session_state.profit_target_sell_percentage,
                    'stop_loss_multiplier': st.session_state.stop_loss_multiplier,
                    'entry_multiplier': st.session_state.entry_multiplier
                    # Add any other parameters needed by your rules
                }
                reset_tx_log()  # Clear previous transactions
                
                with st.spinner("Running backtest..."):
                    # `results` is a dictionary with P&L, trades, etc.
                    results = run_backtest(st.session_state.historical_data, backtest_params)
                    
                    # No need to concat anything into `tx_log`. We already have all the 
                    # trades in st.session_state.tx_log because `log_transaction` appended them.
                    
                    # If you want to store the final results dictionary for display:
                    st.session_state.backtest_results = results

                    # Calculate PnL from DataFrame
                    pnl_results = calculate_trade_pnl_with_percentage(st.session_state.tx_log)
                    st.session_state.backtest_results = pnl_results
                st.success("Backtest completed!")
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                logger.error(f"Backtest failed in UI: {str(e)}")

    # Results display
    if st.session_state.backtest_results:
        logger.info(f"Backtest Results: {st.session_state.backtest_results.keys()}")

        # Highlight Key Stats (First Row)
        st.subheader("🔑 Key Metrics")
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row1_col1.metric("Total P&L $", f"${st.session_state.backtest_results['total_pnl']:,.2f}")
        row1_col2.metric("Avg Return %", f"{st.session_state.backtest_results['avg_return_pct']:.2f}%")
        row1_col3.metric("Overall Return %", f"{st.session_state.backtest_results['overall_return_pct']:.2f}%")

        # Use expanders for detailed trade stats
        with st.expander("📈 Long Trade Stats", expanded=True):
            row2_col1, row2_col2, row2_col3 = st.columns(3)
            row2_col1.metric("Long Trades", f"{st.session_state.backtest_results['trades_long']}")
            row2_col2.metric("Long P&L $", f"${st.session_state.backtest_results['pnl_long']:.2f}")
            row2_col3.metric("Long % Return", f"{st.session_state.backtest_results['return_pct_long']:.2f}%")

        with st.expander("📉 Short Trade Stats", expanded=True):
            row3_col1, row3_col2, row3_col3 = st.columns(3)
            row3_col1.metric("Short Trades", f"{st.session_state.backtest_results['trades_short']}")
            row3_col2.metric("Short P&L $", f"${st.session_state.backtest_results['pnl_short']:.2f}")
            row3_col3.metric("Short % Return", f"{st.session_state.backtest_results['return_pct_short']:.2f}%")

        
        st.subheader("Trade History")
        trades_df = st.session_state.backtest_results["trades"]
        if trades_df.empty:
            st.warning("No trades executed...")
        else:
            st.dataframe(trades_df)




if __name__ == "__main__":
    main()
