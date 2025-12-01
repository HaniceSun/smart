#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')
import pylab as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import yfinance as yf
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from zoneinfo import ZoneInfo
import datetime


class Data2Features():
    def __init__(self):
        self.time_zone = ZoneInfo("America/New_York")
        self.time_format = '%Y-%m-%d %H:%M:%S'

    def download_data(self, symbols='SPY,QQQ,IWM,TLT,^VIX,GLD,CL=F,DX-Y.NYB,EURUSD=X,JPYUSD=X,^TNX,^IRX', data_folder='data', data_file='data.txt', start_date=None, end_date=None, interval='1d', market='stock', timeframe='day', auto_adjust=True, neg_ffill=True, period='max', sleep_time=3, retries=5):

        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        symbols = sorted([x.strip() for x in symbols.split(',')])

        success = {}
        for symbol in symbols:
            for n in range(1, 1 + retries):
                try:
                    self.log(f'downloading {symbol}')
                    ouF = f'{data_folder}/{symbol.replace("/", ".")}.txt.gz'

                    if start_date is None and end_date is None:
                        df = yf.download(symbol, interval=interval, period=period, auto_adjust=auto_adjust)
                    elif start_date and end_date:
                        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=auto_adjust)
                    elif start_date:
                        df = yf.download(symbol, start=start_date, interval=interval, period=period, auto_adjust=auto_adjust)
                    elif end_date:
                        df = yf.download(symbol, end=end_date, interval=interval, period=period, auto_adjust=auto_adjust)

                    df.columns = [x[0] for x in df.columns]
                    df['datetime'] = df.index
                    df.to_csv(ouF, header=True, index=False, sep='\t')
                    success[symbol] = True
                    break
                except Exception as e:
                    self.log(f'downloading {symbol} attempt {n} {e}')
                    time.sleep(sleep_time * n)
                time.sleep(sleep_time)

        if len(success) != len(symbols):
            self.log('error: downloadig symbols not completed')
        else:
            self.log(f'merging data to {data_file}')
            L = []
            for symbol in symbols:
                inF = f'{data_folder}/{symbol.replace("/", ".")}.txt.gz'
                df = pd.read_table(inF, header=0, sep='\t')
                df2 = df[['datetime', 'Close']]
                df2.columns = ['datetime', symbol]
                L.append(df2)

            df = L[0]
            for x in L[1:]:
                df = pd.merge(df, x, on='datetime')

            if neg_ffill:
                for col in df.columns:
                    if col not in ['datetime']:
                        df[col] = df[col].mask(df[col] < 0).ffill()
            else:
                wh = [True if col not in ['datetime'] else False for col in df.columns]
                wh2 = (df.loc[:, wh] > 0).all(axis=1)
                n_neg = df.shape[0] - sum(wh2)
                if n_neg:
                    self.log(f'removed {n_neg} rows with negative values')
                    print(df.loc[~wh2, ])
                df = df.loc[wh2, ]

            df.to_csv(data_file, header=True, index=False, sep='\t')


    def get_log_return_labels(self, target_symbol='SPY', horizon=5, delta=-0.01, delta_scale='linear', data_file='data.txt', labeling=True):
        data_file2 = data_file.replace(".txt", "_LogReturn.txt")
        df = pd.read_table(data_file, header=0, sep='\t')

        #--- remove rows with negative values
        wh = [True if col not in ['datetime'] else False for col in df.columns]
        wh2 = (df.loc[:, wh] > 0).all(axis=1)
        n_neg = df.shape[0] - sum(wh2)

        if n_neg:
            self.log(f'removed {n_neg} rows with negative values')
            print(df.loc[~wh2, ])
        df = df.loc[wh2, ]
        #---

        self.log('calculating log10 return')
        df_ret = pd.DataFrame()
        for x in df.columns:
            if x in ['datetime']:
                df_ret[x] = df[x]
            else:
                df_ret[f'LogReturn_{x}'] = np.log10(df[x]) - np.log10(df[x].shift(1))

        log_ret = df_ret[f'LogReturn_{target_symbol}'].shift(-horizon).rolling(horizon).sum()
        ret = np.power(10, log_ret)

        if labeling:
            self.log('labeling each sample based on the delta')
            if delta:
                if delta_scale == 'log10':
                    if delta < 0:
                        df_ret['class'] = (log_ret < 1 + delta).astype(int)
                    elif delta > 0:
                        df_ret['class'] = (log_ret > delta).astype(int)
                elif delta_scale == 'linear':
                    if delta < 0:
                        df_ret['class'] = (ret < 1 + delta).astype(int)
                    elif delta > 0:
                        df_ret['class'] = (ret > delta).astype(int)
            else:
                self.log("warning: delta shouldn't be zero")


            n_positive = df_ret['class'].sum()
            n_total = df_ret.shape[0]
            self.log(f'{n_positive} positive cases in {n_total} samples ({n_positive/n_total*100:.2f}%)')
        else:
            df_ret['class'] = '.'

        df_ret.dropna(inplace=True)
        df_ret.to_csv(data_file2, header=True, index=False, sep='\t')


    def make_features(self, data_file='data_LogReturn.txt', drop_ffill_na=True, test_data_start_date='2024-01-01', target_symbol='SPY', split=True):
        data_file2 = data_file.replace('.txt', '_Features.txt')

        self.df = pd.read_table(data_file, header=0, sep='\t')
        print(self.df)

        self.features = {}
        self.features['datetime'] = self.df['datetime']
        self.features['class'] = self.df['class']

        self.score_momentum_volatility()
        self.score_momentum_skewness()
        self.score_momentum_kurtosis()
        self.score_momentum_shannon_entropy()
        self.score_mean_reversion_hurst_exponent()
        self.score_cross_asset_relation_rollingOLS(target_symbol=target_symbol)
        self.score_cross_asset_relation_correlation()
        self.score_information_theory()

        self.df = pd.DataFrame(self.features)

        if drop_ffill_na:
            try:
                idx = self.df[self.df.notna().all(axis=1)].index[0]
                self.df = self.df.loc[idx:].ffill()
            except Exception as e:
                self.log(f'making features {e}')

        self.df.to_csv(data_file2, header=True, index=False, sep='\t')
        self.log(f'saved features to {data_file2}')

        if split:
            self.train_test_split(test_data_start_date=test_data_start_date, data_file=data_file2)
            self.log(f'split dataset to train and split on date {test_data_start_date}')

    def train_test_split(self, test_data_start_date='2024-01-01', data_file='data_LogReturn_Features.txt'):
        ouF_train = data_file.replace('.txt', '_train.txt')
        ouF_test = data_file.replace('.txt', '_test.txt')
        df = pd.read_table(data_file, header=0, sep='\t')
        wh = df['datetime'] < test_data_start_date
        self.df_train = df.loc[wh]
        self.df_test = df.loc[~wh]
        self.df_train.to_csv(ouF_train, header=True, index=False, sep='\t')
        self.df_test.to_csv(ouF_test, header=True, index=False, sep='\t')

    def explore_features(self, variance_threshold=1e-4, correlation_threshold=0.95, show_top_n=20, fontsize=3, data_file='data_LogReturn_Features_train.txt'):
        '''
        Filter features with low variance, high correlation, low mutual information (MI) scores,
        and then do standardization within each training fold during cross-validation to avoid data leakage.
        Here, however, we are exploring whether it is true that the features with the highest MI scores
        come primarily from the Hurst exponents of cross-asset classes, rather than from SPY itself.
        '''

        self.df = pd.read_table(data_file, header=0, sep='\t') 

        wh = np.array([True if x not in ['datetime', 'class'] else False for x in self.df.columns])
        X = self.df.loc[:, wh]
        y = self.df['class'].values
        self.log(f'exploring features, {X.shape[1]} features before filtering')
        X = self.remove_low_variance(X, threshold=variance_threshold)
        self.log(f'exploring features, {X.shape[1]} features after removing low variance')
        X = self.remove_high_corr(X, threshold=correlation_threshold)
        self.log(f'exploring features, {X.shape[1]} features after removing high correlation')

        df = self.sort_mi_scores(X, y)
        ouF = data_file.replace('.txt', '_MI.txt')
        ouF2 = data_file.replace('.txt', '_MI.png')
        df.to_csv(ouF, header=False, index=False, sep='\t')
        self.log('exploring features, top 20 features sorted by MI scores')
        print(df.head(show_top_n))

        fig = plt.figure()
        ax = fig.add_subplot()
        sns.barplot(x='features', y='MI', data=df, ax=ax)
        plt.xticks(rotation=90, fontsize=fontsize)
        ax.set_xlabel('')
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(ouF2)

    def score_momentum_volatility(self, ws=[21, 63]):
        for w in ws:
            for col in self.df.columns:
                if col.startswith('LogReturn'):
                    feature = col.replace('LogReturn', f'Volatility_{w}')
                    self.features[feature] = self.df[col].rolling(w).std()

    def score_momentum_skewness(self, ws=[21, 63]):
        for w in ws:
            for col in self.df.columns:
                if col.startswith('LogReturn'):
                    feature = col.replace('LogReturn', f'Skewness_{w}')
                    self.features[feature] = self.df[col].rolling(w).skew()

    def score_momentum_kurtosis(self, ws=[21, 63]):
        for w in ws:
            for col in self.df.columns:
                if col.startswith('LogReturn'):
                    feature = col.replace('LogReturn', f'Kurtosis_{w}')
                    self.features[feature] = self.df[col].rolling(w).kurt()

    def score_momentum_shannon_entropy(self, bins=30):
        for col in self.df.columns:
            if col.startswith('LogReturn'):
                feature = col.replace('LogReturn', f'Shannon_{bins}')
                self.features[feature] = self._shannon_entropy(self.df[col], bins=bins)

    def score_mean_reversion_hurst_exponent(self, ws=[16, 64, 256]):
        for col in self.df.columns:
            if col.startswith('LogReturn'):
                hurst_dict = self._hurst_rs_vectorized(self.df[col], ws)
                for w in ws:
                    feature = col.replace('LogReturn', f'Hurst_{w}')
                    self.features[feature] = hurst_dict[w]

    def score_cross_asset_relation_rollingOLS(self, target_symbol='SPY', ws=[21, 63]):
        for w in ws:
            for col in self.df.columns:
                if col.startswith('LogReturn'):
                    if not col.endswith(target_symbol):
                        feature = col.replace('LogReturn', f'RollingOLS_{w}')
                        self.features[feature] = self._rolling_ols(self.df[col], self.df[f'LogReturn_{target_symbol}'], w=w)

    def score_cross_asset_relation_correlation(self, ws=[21, 63]):
        for w in ws:
            for col in self.df.columns:
                if col.startswith('LogReturn'):
                    feature = col.replace('LogReturn', f'Correlation_{w}')
                    self.features[feature] = self.df[col].rolling(w).corr()

    def score_information_theory(self, ws=[21, 126], bins=30):
        for col in self.df.columns:
            if col.startswith('LogReturn'):
                feature = col.replace('LogReturn', f'InfoTheory_{ws[0]}-{ws[1]}')
                self.features[feature] = self._rolling_kl_short_long(self.df[col], short_window=ws[0], long_window=ws[1], bins=30)

    def _shannon_entropy(self, x, bins=30):
        hist, _ = np.histogram(x, bins=bins, density=True)
        p = hist[hist > 0]
        return -(p * np.log(p)).sum()

    def _hurst_rs_vectorized(self, logret: pd.Series, windows=[16, 64, 256]):
        """
        Compute Hurst exponent over log returns using R/S statistic.
        logret: pandas Series of log returns
        """
        ret_arr = logret.values
        n = len(ret_arr)
        stride = ret_arr.strides[0]
        
        hurst = {}
    
        for tau in windows:
            if tau > n:
                hurst[tau] = pd.Series(np.nan, index=logret.index)
                continue
    
            shape = (n - tau + 1, tau)
            strides = (stride, stride)
            windows_arr = np.lib.stride_tricks.as_strided(
                ret_arr, shape=shape, strides=strides
            )
    
            # Mean-adjusted
            dm = windows_arr - windows_arr.mean(axis=1, keepdims=True)
    
            # Cumulative sum → range R
            cum = dm.cumsum(axis=1)
            R = cum.max(axis=1) - cum.min(axis=1)
    
            # Standard deviation
            S = windows_arr.std(axis=1, ddof=1)
    
            # ---- Safe division: only compute on S > 0 ---- #
            RS = np.full_like(S, np.nan)
            valid = (S > 0) & (R > 0)
            RS[valid] = R[valid] / S[valid]
    
            # Hurst exponent
            H = np.full_like(RS, np.nan)
            valid_H = RS > 0
            H[valid_H] = np.log(RS[valid_H]) / np.log(tau)
    
            # Align with original index
            aligned = pd.Series(
                np.r_[np.full(tau - 1, np.nan), H],
                index=logret.index
            )
    
            hurst[tau] = aligned

        return hurst

    def _rolling_ols(self, x, y, w):
        betas = np.full(len(y), np.nan)
        intercepts = np.full(len(y), np.nan)

        for i in range(w, len(y)):
            yi = y[i-w:i]
            xi = x[i-w:i]
            A = np.column_stack([np.ones(w), xi])
            beta = np.linalg.lstsq(A, yi, rcond=None)[0]
            intercepts[i], betas[i] = beta
        return betas

    def _kl_divergence(self, p, q, epsilon=1e-10):
        p = np.asarray(p, dtype=float) + epsilon
        q = np.asarray(q, dtype=float) + epsilon
        return np.sum(p * np.log(p / q))

    def _rolling_kl_short_long(self, returns, short_window=21, long_window=126, bins=30):
        kl_values = pd.Series(index=returns.index, dtype=float)

        for i in range(long_window, len(returns)):
            short_slice = returns.iloc[i-short_window:i]
            long_slice = returns.iloc[i-long_window:i]

            hist_short, _ = np.histogram(short_slice, bins=bins, density=True)
            hist_long, _ = np.histogram(long_slice, bins=bins, density=True)

            hist_short /= hist_short.sum()
            hist_long /= hist_long.sum()

            kl_values.iloc[i] = self._kl_divergence(hist_short, hist_long)

        return kl_values

    def remove_low_variance(self, X, threshold=1e-3):
        selector = VarianceThreshold(threshold=threshold)
        X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
        return X

    def remove_high_corr(self, X, threshold=0.95):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
        X.drop(columns=to_drop, inplace=True)
        return X

    def sort_mi_scores(self, X, y, discrete_features='auto', random_state=42):
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
        df = pd.DataFrame()
        df['features'] = X.columns
        df['MI'] = mi
        df.sort_values(by='MI', ascending=False, inplace=True)
        return df

    def standardize_features(self, X_train, X_val=None, scaler=None):
        if not scaler:
            scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        if X_val is not None:
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        return X_train, X_val, scaler

    def log(self, txt=''):
        current_time = datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        print(f'{current_time} {txt}', flush=True)

if __name__ == '__main__':
    d2f = Data2Features()
    d2f.download_data()
    d2f.get_log_return_labels()
    d2f.make_features()
    d2f.train_test_split()
    d2f.explore_features()
