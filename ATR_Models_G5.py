from pydantic import BaseModel
from datetime import date
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd


class Candle(BaseModel):
    cdate: date
    open: float
    high: float
    low: float
    close: float

class MyTransaction:
    def __init__(self):
        self.atr_period: int = 20
        self.k: float = 2 
        self.point_value: float = 5
        self.sma_window: int = 3 
        self.span: int = 20
        self.atr_channel: Optional[pd.DataFrame] = None
        self.atr_values: List[Tuple[date, Optional[float]]] = []
        self.buy_prices = []
        self.candlesticks: List[Dict] = []
        self.current_position: Optional[str] = None
        self.cut_loss: list[Dict] = []
        self.ema_20: Optional[pd.DataFrame] = None
        self.entry_signals: list[Dict] = []
        self.Entry_submit_order: list[dict] = {}
        self.new_day = [] 
        self.sell_prices = []
        self.signal: str = 'no' 
        self.signals: List[Dict] = []   
        self.sma: Optional[pd.DataFrame] = None 
        self.submit_order: list[dict] = {}
        self.take_profit: list[Dict] = []
        self.tick = []
        self.trend_cache: Dict[date, str] = {}
        self.trend: str = "unknown"
        self.tp = []  
 
    def add_candle(self, candle: Candle):
        if not isinstance(candle, Candle):
            self.logger.error("Invalid candle data")
            return
        self.candlesticks.append(candle.dict())

    def update_ema_20(self, ema_20: pd.DataFrame):
        self.ema_20 = ema_20

    def add_atr_value(self, date: date, atr_value: Optional[float]):
        self.atr_values.append((date, atr_value))

    def update_trend_cache(self, date: date, trend: str):
        self.trend_cache[date] = trend

    def set_trend(self, trend: str):
        self.trend = trend


    def update_entry_signal(self, date: date, signal: Optional[str]):
        existing_signal = next((s for s in self.entry_signals if s['date'] == date), None)
        if existing_signal:
            existing_signal['signal'] = signal
        else:
            self.entry_signals.append({'date': date, 'signal': signal})
  
    def update_TP_signal(self, date: date, signal: Optional[str]):
        existing_signal = next((s for s in self.take_profit if s['date'] == date), None)
        
        if existing_signal:
            if existing_signal['signal'] not in ["short take profit", "long take profit"]:
                existing_signal['signal'] = signal
        else:
            self.take_profit.append({'date': date, 'signal': signal})
        
        if signal in ["short take profit", "long take profit"]:
            for s in self.take_profit:
                if s['date'] > date and s['signal'] not in ["short take profit", "long take profit"]:
                    s['signal'] = signal


    def update_CL_signal(self, date: date, signal: Optional[str]):
        existing_signal = next((s for s in self.cut_loss if s['date'] == date), None)
        if existing_signal:
            existing_signal['signal'] = signal
        else:
            self.cut_loss.append({'date': date, 'signal': signal})
   
    def update_transaction(self, transac_timestamp, transaction):
           if date not in self.submit_order:
               self.submit_order[transac_timestamp] = []  
           self.submit_order[transac_timestamp].append(transaction) 
           
    def add_transaction(self, transac_timestamp, transaction):
            if date not in self.Entry_submit_order:
                self.Entry_submit_order[transac_timestamp] = []  
            self.Entry_submit_order[transac_timestamp].append(transaction) 
    
    def add_buy_price(self, price):
        self.buy_prices.append(price)
        
    def add_sell_price(self, price):
        self.sell_prices.append(price)
  
            
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('algo.log', mode='w')]
)

logger = logging.getLogger('algo')
