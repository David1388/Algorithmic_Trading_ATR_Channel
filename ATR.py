from ATR_Models_G5 import MyTransaction
from ATR_Models_G5 import Candle, logger
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import brokerage


def make_connection(fname):
    username = "something"
    password = "something"
    conn = brokerage.login("my_brokerage", username, password, speed=0.2)
    instrument = fname
    conn.connect("KLSE", instrument)
    return conn


def make_candlesticks(tick, txn):
    timestamp, price = tick
    date = timestamp.date()
    txn.tick.append((date, price))
    new_day = len(
        txn.candlesticks) == 0 or txn.candlesticks[-1]['cdate'] != date
   
    if new_day:
        new_candle = Candle(
            cdate=timestamp.date(),
            open=price,
            high=price,
            low=price,
            close=price
        )
        txn.candlesticks.append(new_candle.dict())
        logger.info(f"New candle created: {new_candle.dict()}")
    else:
        current_candle = txn.candlesticks[-1]
        current_candle['high'] = max(current_candle['high'], price)
        current_candle['low'] = min(current_candle['low'], price)
        current_candle['close'] = price

def calculate_atr(txn):
    if len(txn.candlesticks) < txn.atr_period:
        logger.info("Cannot calculate ATR")

        last_date = pd.to_datetime(
            txn.candlesticks[-1]['cdate']).date() if txn.candlesticks else None
        return last_date, None

    df = pd.DataFrame(txn.candlesticks)
    df['cdate'] = pd.to_datetime(df['cdate']).dt.date

    df_last = df.groupby('cdate').last().reset_index()

    df_last['high_low'] = df_last['high'] - df_last['low']
    df_last['high_close_prev'] = abs(
        df_last['high'] - df_last['close'].shift())
    df_last['low_close_prev'] = abs(df_last['low'] - df_last['close'].shift())
    df_last['true_range'] = df_last[['high_low',
                                     'high_close_prev', 'low_close_prev']].max(axis=1)

    df_last['atr'] = df_last['true_range'].rolling(
        window=txn.atr_period).mean()

    atr_values = df_last[['cdate', 'atr']].to_records(index=False)

    txn.atr_values.clear()

    for date, atr_value in atr_values:
        txn.atr_values.append((date, atr_value))

    if txn.atr_values:
        last_atr_value = txn.atr_values[-1]
    else:
        last_atr_value = (df_last['cdate'].max()
                          if not df_last.empty else None, None)

    logger.info(f"ATR: {last_atr_value[1]}, Date: {last_atr_value[0]}")

    return last_atr_value


def SMA(txn):

    window = txn.sma_window

    if not txn.candlesticks:
        logger.info("No candlestick data available for SMA calculation.")
        return None

    df = pd.DataFrame(txn.candlesticks)
    df['cdate'] = pd.to_datetime(df['cdate']).dt.date
    df = df.set_index('cdate')

    if len(df) < window:
        logger.info(f"Not enough data to calculate {window}-day SMA.")
        return None

    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

    sma_df = df[['close', f'sma_{window}']].reset_index()

    txn.sma = sma_df

    

    return sma_df




def calculate_moving_averages(txn):
    if not txn.candlesticks:
        logger.info("No candlestick data available for EMA calculation.")
        txn.ema_20 = None
        return

    df = pd.DataFrame(txn.candlesticks)
    df['cdate'] = pd.to_datetime(df['cdate']).dt.date
    df = df.set_index('cdate')

    if len(df) < txn.span:
        logger.info("Not enough data to calculate 20-day EMA.")
        txn.ema_20 = None
        return

    prices = pd.Series(df['close'])

    span = txn.span
    alpha = txn.k / (span + 1)

    ema = pd.Series([None] * len(prices), index=prices.index)
    ema.iloc[span - 1] = prices[:span].mean()

    for i in range(span, len(prices)):
        ema.iloc[i] = (prices.iloc[i] * alpha) + (ema.iloc[i - 1] * (1 - alpha))

    ema_df = pd.DataFrame({
        'cdate': prices.index,
        'ema_20': ema
    }).reset_index(drop=True)

    txn.ema_20 = ema_df

    


def calculate_atr_channel(txn):
    last_atr_value = calculate_atr(txn)
    if last_atr_value[1] is None:
        logger.info("ATR value is None, cannot calculate ATR Channel.")
        return pd.DataFrame()

    df = pd.DataFrame(txn.ema_20)
    df['cdate'] = pd.to_datetime(df['cdate']).dt.date
    df = df.set_index('cdate')

    atr_df = pd.DataFrame(txn.atr_values, columns=['cdate', 'atr'])
    atr_df['cdate'] = pd.to_datetime(atr_df['cdate']).dt.date
    atr_df = atr_df.set_index('cdate')

    df = df.join(atr_df)

    df['upper_channel'] = df['ema_20'] + txn.k * df['atr']
    df['lower_channel'] = df['ema_20'] - txn.k * df['atr']

    txn.atr_channel = df[['ema_20', 'upper_channel', 'lower_channel']].reset_index()

    logger.info(f"Upper Channel:\n{txn.atr_channel[['cdate', 'upper_channel']].tail(1)}")
    logger.info(f"Lower Channel:\n{txn.atr_channel[['cdate', 'lower_channel']].tail(1)}")

    return txn.atr_channel

    
def identify_trend(txn):
    if txn.atr_channel is None or txn.sma is None:
        txn.set_trend("unknown")
        logger.info("ATR channel or SMA none.")
        return txn.trend

    df_candlesticks = pd.DataFrame(txn.candlesticks)
    df_last = df_candlesticks.groupby('cdate').last().reset_index()

    current_date = df_last['cdate'].iloc[-1]

    sma_latest = txn.sma['sma_3'].iloc[-1]  

    upper_channel = txn.atr_channel['upper_channel'].iloc[-1]
    lower_channel = txn.atr_channel['lower_channel'].iloc[-1]

    if sma_latest > upper_channel:
        txn.set_trend("overbought")
    elif sma_latest < lower_channel:
        txn.set_trend("oversold")
    else:
        txn.set_trend("neutral")

    txn.update_trend_cache(current_date, txn.trend)

    logger.info(f" {current_date}: {txn.trend}")
    return txn.trend

def determine_entry_signal(txn):
    if len(txn.trend_cache) < 2:

        return

    dates = list(txn.trend_cache.keys())
    yesterday_date = dates[-2]
    today_date = dates[-1]
    yesterday_trend = txn.trend_cache[yesterday_date]
    today_trend = txn.trend_cache[today_date]

    signal_type = None
    if txn.current_position == 0:
        if yesterday_trend == 'overbought' and today_trend == 'neutral' :
            signal_type = 'short'
            logger.info(f" Entry signal{signal_type} generated for {today_date}")
        elif yesterday_trend == 'oversold' and today_trend == 'neutral' :
            signal_type = 'long'
            logger.info(f" Entry signal{signal_type} generated for {today_date}")
        else:
            logger.info(f"No Entry signal generated for {today_date}")
    else:
        logger.info("position not 0")
    txn.update_entry_signal(today_date, signal_type)


def determine_Take_profit_signal(txn):
    if txn.ema_20 is None or len(txn.candlesticks) == 0:
        logger.info("EMA data or candlestick data is missing.")
        return

    last_candle = txn.candlesticks[-1]
    today_date = last_candle['cdate']
    close_price = last_candle['close']

    if txn.tp:
        last_ema = txn.tp[0]['price']
    else:
        last_ema = None

    take_profit_type = None
    if last_ema is not None:
        if txn.current_position == 1 and close_price >= last_ema:
            take_profit_type = 'long take profit'
            logger.info(f"TP_CLOSE_long {today_date} (Close Price: {close_price}, EMA(20): {last_ema})")
        elif txn.current_position == -1 and close_price <= last_ema:
            take_profit_type = 'short take profit'
            logger.info(f"TP_CLOSE_SHORT {today_date} (Close Price: {close_price}, EMA(20): {last_ema})")
        else:
            logger.info(f"No take profit signal generated for {today_date} (Close Price: {close_price}, EMA(20): {last_ema})")
    else:
        logger.info("No valid EMA available for take profit calculation.")

    txn.update_TP_signal(today_date, take_profit_type)



def determine_Cut_Loss_signal(txn):

    if txn.candlesticks is None or txn.atr_channel is None:
        logger.info("candlesticks or atr_channel is missing or None")
        return

    last_candle = txn.candlesticks[-1]
    today_date = last_candle['cdate']
    close_price = last_candle['close']

    upper_channel = txn.atr_channel['upper_channel'].iloc[-1]
    lower_channel = txn.atr_channel['lower_channel'].iloc[-1]

    cut_loss_type = None
    
    if txn.current_position == 1 and close_price < lower_channel:
        cut_loss_type = 'long cut loss'
        logger.info("CL_CLOSE_Long")
    elif txn.current_position == -1 and close_price > upper_channel:
        cut_loss_type = 'short cut loss'
        logger.info("CL_CLOSE_SHORT")
    else:
        logger.info(f"No Cut loss signal generated for {today_date}")

    txn.update_CL_signal(today_date, cut_loss_type)
   
def do_entry_signal(txn, conn, tick):
    timestamp, price = tick
    date = timestamp.date()
    new_day = False
    if len(txn.tick) > 1:
        new_day = len(txn.tick) == 0 or txn.tick[-2][0] != date
        logger.info(f"Processing tick for date: {date}")
    elif len(txn.tick) == 1:
        
        logger.info("not enough tick")
    else:
        logger.warning(f"no new day in do enrty for {date}")
   

    if new_day:
        prev_signal = txn.entry_signals[-2]['signal'] if len(txn.entry_signals) > 1 else None
        
        if prev_signal == 'long':
            logger.info(f"Executing 'long' entry signal based on previous day's signal for {date}.")
            conn.submit_order(price=1, lot_size=1, order_type='Market', action='Buy')
        elif prev_signal == 'short':
            logger.info(f"Executing 'short' entry signal based on previous day's signal for {date}.")
            conn.submit_order(price=1, lot_size=1, order_type='Market', action='Sell')
        else:
            logger.warning(f"No valid signal to execute for {date}.")
    else:
        logger.info(f"No new day. Skipping order placement for date {date}.")


def do_Take_Profit_signal(txn, conn, tick): 
    if not txn.take_profit:
        logger.info("No signal to execute Take profit.")
        return
    timestamp, price = tick
    date = timestamp.date()
    new_day = False
    if len(txn.tick) > 1:
        new_day = len(txn.tick) == 0 or txn.tick[-2][0] != date
        logger.info(f"Processing tick for date: {date}")
    elif len(txn.tick) == 1:
        
        logger.info("not enough tick")
    else:
        logger.warning(f"no new day in do enrty for {date}")
    prev_signal = txn.entry_signals[-2]['signal'] if len(txn.entry_signals) > 1 else None
    
    signal = txn.take_profit[-1]['signal'] if txn.take_profit else None
    date = txn.take_profit[-1]['date']
    if len(txn.ema_20) > 19:
        EMA = txn.ema_20.iloc[-1]['ema_20']
        logger.info(f"EMA {EMA}.")
    if new_day:
        if prev_signal == 'short':
            logger.info(f"Executing TP_CLOSE_SHORT on {date}.")
            conn.submit_order(price= EMA,  
                              lot_size=1,
                              order_type='Limit',  
                              action='Buy')
            txn.tp.clear()
            txn.tp.append({'date': date, 'price': EMA,})
        elif  prev_signal == 'long':
            logger.info(f"Executing TP_CLOSE_long on {date}.")
            conn.submit_order(price= EMA,  
                              lot_size=1,
                              order_type='limit', 
                              action='Sell') 
            txn.tp.clear()
            txn.tp.append({'date': date, 'price': EMA,})
        else:
            logger.warning(f"Unrecognized signal type: {signal}")
    else:
        logger.info("not enough EMA")

def do_Cut_Loss_signal(txn, conn, tick,p):
    timestamp, price = tick
    date = timestamp.date()
    new_day = False
    if len(txn.tick) > 1:
         new_day = len(txn.tick) == 0 or txn.tick[-2][0] != date
         logger.info(f"Processing tick for date: {date}")
    elif len(txn.tick) == 1:
         logger.info("not enough tick")
    else:
         logger.warning(f"no new day in do enrty for {date}")
    
    if new_day:
        prev_cut_loss_signal = txn.cut_loss[-2]['signal'] if len(txn.cut_loss) > 1 else None
        if prev_cut_loss_signal == 'long cut loss':
            logger.info(f"Executing 'CL_CLOSE_LONG' cut loss signal for {date}.")
            conn.submit_order(price=1, lot_size=1,
                              order_type='Market',
                              action='Sell')  
            conn.cancel_order(p[0]['ticket_no'])
        elif prev_cut_loss_signal == 'short cut loss':
            logger.info(f"Executing 'CL_CLOSE_SHORT' cut loss signal for {date}.")
            conn.submit_order(price=1, lot_size=1,
                              order_type='Market',
                              action='Buy') 
            conn.cancel_order(p[0]['ticket_no'])
        else:
            logger.warning(f"Unknown cut loss signal '{prev_cut_loss_signal}' for {date}.")
    else:
        logger.info(f"No new day. Skipping order placement for date {date}.")
        
def update_position(txn, conn):
    pending, transac, cancel = conn.get_transactions()

    long_positions = 0
    short_positions = 0

    for i, transaction in enumerate(transac):
        if transaction['action'] == 'Buy':
            long_positions += 1
        elif transaction['action'] == 'Sell':
            short_positions += 1
        
        transac_date = transaction['transac_timestamp'].date()
        txn.update_transaction(transac_date, transaction)
        
        if i % 2 == 0:
            txn.add_transaction(transac_date, transaction)
 

    txn.current_position = long_positions - short_positions  
    logger.info(f"long_positions: {long_positions}")
    logger.info(f"short_positions: {short_positions}")
    logger.info(f"positions: {txn.current_position}")


def calculate_pnl(txn, t):
    total_pnl = 0.0

    for trans in t:
        if trans['action'] == 'Buy':
            txn.buy_prices.append(trans['transac_price'])
        elif trans['action'] == 'Sell':
            if txn.buy_prices:
                buy_price = txn.buy_prices.pop(0)  
                sell_price = trans['transac_price']
                pnl = sell_price - buy_price  
                total_pnl += pnl
                logger.info(f"trans: buy price {buy_price:.2f}, sell price {sell_price:.2f}, pnl {pnl:.2f}")
            else:
                logger.warning("No buy price available for sell action")


    total_pnl = round(total_pnl, 2)
    txn.total_pnl = total_pnl*txn.point_value
    logger.info(f"Total P&L: {total_pnl:.2f}")

    return total_pnl



def visualize_position_and_pnl(txn):
    df = pd.DataFrame(txn.candlesticks)
    df['timestamp'] = pd.to_datetime(df['cdate'])  
    df.set_index('timestamp', inplace=True)

    atr_channel_df = calculate_atr_channel(txn)
    SMA_3 = SMA(txn)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    for idx, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1.5)
        ax1.add_patch(plt.Rectangle((idx - pd.Timedelta(minutes=700), min(row['open'], row['close'])),
                                    pd.Timedelta(minutes=1400), abs(row['close'] - row['open']),
                                    color=color, linewidth=0, alpha=0.7))
    
    if atr_channel_df is not None and not atr_channel_df.empty:
        df = df.join(atr_channel_df.set_index('cdate'))
        ax1.plot(df.index, df['upper_channel'], color='green', linestyle='--', label='Upper Channel')
        ax1.plot(df.index, df['ema_20'], color='blue', linestyle='--', label='EMA_20')
        ax1.plot(df.index, df['lower_channel'], color='red', linestyle='--', label='Lower Channel')
    else:
        logger.warning("ATR channel dataframe is empty or None")

    if SMA_3 is not None and not SMA_3.empty:
        df = df.join(SMA_3.set_index('cdate'), rsuffix='_sma')  
        ax1.plot(df.index, df['sma_3'], color='purple', linestyle='--', label='SMA_3')
    else:
       logger.warning("SMA dataframe is empty or None")

    if txn.entry_signals:
        entry_df = pd.DataFrame(txn.entry_signals, columns=['date', 'signal'])
        entry_df['date'] = pd.to_datetime(entry_df['date'])
        
        entry_df = entry_df.join(df[['close']], on='date', how='left')

        if 'close' not in entry_df.columns:
            logger.warning("Error: 'close' column not found in entry_df after join.")
            return
        
        buy_signals = entry_df[entry_df['signal'] == 'long']
        sell_signals = entry_df[entry_df['signal'] == 'short']

        ax1.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='yellow', label='Long Signals', s=200)
        for idx, entry in buy_signals.iterrows():
            ax1.annotate(
                f"Long @ {entry['date'].strftime('%Y-%m-%d')}",
                (entry['date'], entry['close']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='Black'
            )

        ax1.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='orange', label='Short Signals', s=200)
        for idx, entry in sell_signals.iterrows():
            ax1.annotate(
                f"Short @ {entry['date'].strftime('%Y-%m-%d')}",
                (entry['date'], entry['close']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='Black'
            )
    if txn.cut_loss:
      cut_loss_df = pd.DataFrame(txn.cut_loss)
      cut_loss_df['date'] = pd.to_datetime(cut_loss_df['date'])

      cut_loss_df = cut_loss_df.join(df[['close']], on='date', how='left')
  
      if 'close' not in cut_loss_df.columns:
          print("Error: 'close' column not found in cut_loss_df after join.")
          return
 
      cut_loss_df = cut_loss_df[cut_loss_df['signal'].isin(['short cut loss', 'long cut loss'])]
      
      ax1.scatter(cut_loss_df['date'], cut_loss_df['close'], marker='x', color='purple', label='Cut Loss Signals', s=200)
      for idx, entry in cut_loss_df.iterrows():
          ax1.annotate(
              f"{entry['signal']} @ {entry['date'].strftime('%Y-%m-%d')}",
              xy=(entry['date'], entry['close']),
              textcoords="offset points",
              xytext=(0, 10),
              ha='center',
              color='Black'
          )
    if txn.take_profit:
       take_profit_df = pd.DataFrame(txn.take_profit)
       take_profit_df['date'] = pd.to_datetime(take_profit_df['date'])
       
       take_profit_df = take_profit_df.join(df[['close']], on='date', how='left')
    
       if 'close' not in take_profit_df.columns:
           print("Error: 'close' column not found in take_profit_df after join.")
           return
       
       take_profit_df = take_profit_df[take_profit_df['signal'].isin(['short take profit', 'long take profit'])]
       
       ax1.scatter(take_profit_df['date'], take_profit_df['close'], marker='s', color='cyan', label='Take Profit Signals', s=200)
       for idx, entry in take_profit_df.iterrows():
           ax1.annotate(
               f"{entry['signal']} @ {entry['date'].strftime('%Y-%m-%d')}",
               xy=(entry['date'], entry['close']),
               textcoords="offset points",
               xytext=(0, 10),
               ha='center',
               color='Black'
           )

    ax1.plot([], [], ' ', label=f'Current Position: {txn.current_position}')
    ax1.plot([], [], ' ', label=f'Total P&L: {txn.total_pnl:.2f}')

    ax1.legend(loc='upper left')
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    ax1.grid(True)
    ax1.legend()

    plt.tight_layout()
    plt.show()

def main(DJF5TO9):
    txn = MyTransaction()
    conn = make_connection(DJF5TO9)
    p,t,c=conn.get_transactions()
    
    for tick in conn.data_stream():
        make_candlesticks(tick, txn)
        calculate_atr(txn)
        SMA(txn)
        calculate_moving_averages(txn)
        calculate_atr_channel(txn)
        identify_trend(txn)
        determine_entry_signal(txn)
        determine_Take_profit_signal(txn)
        determine_Cut_Loss_signal(txn)
        update_position(txn,conn)
        do_entry_signal(txn,conn,tick)
        do_Take_Profit_signal(txn, conn, tick)
        do_Cut_Loss_signal(txn, conn, tick,p)
        calculate_pnl(txn,t)
        visualize_position_and_pnl(txn)

    return txn


if __name__ == "__main__":
   txn = main("data/DJF5TO9")