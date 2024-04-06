import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SMABacktester():
    def __init__(self, symbol, SMA_S, SMA_L, start, end):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.results = None
        self.get_data()
        
    def get_data(self):
        df = yf.download(self.symbol, self.start, self.end)
        data = df.Close.to_frame()
        data["returns"] = np.log(data.Close.div(data.Close.shift(1)))
        data["SMA_S"] = data.Close.rolling(window=int(self.SMA_S)).mean()
        data["SMA_L"] = data.Close.rolling(window=int(self.SMA_L)).mean()
        self.data2 = data
        
        return data
    
    def test_results(self):
        data = self.data2.copy().dropna()
        data["position"] = np.where(data["SMA_S"]>data["SMA_L"], 1, -1)
        data["strategy"] = data["returns"] * data["position"].shift(1)
        data.dropna(inplace=True)
        data["bhreturns"] = data["returns"].cumsum().apply(np.exp)
        data["strategyreturns"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["strategyreturns"].iloc[-1] # Get last row
        outperf = perf - data["bhreturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        if self.results is None:
            print("No results to plot. Run the test first pls! :)")
        else:
            title="{}| SMA_S={} | SMA_L{}".format(self.symbol,self.SMA_S, self.SMA_L)
            self.results[["bhreturns", "strategyreturns"]].plot(figsize=(15,8), fontsize=13, title=title)