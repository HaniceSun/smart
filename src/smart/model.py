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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from itertools import product
import yaml
import json
import ast
import joblib
from zoneinfo import ZoneInfo
import datetime
from importlib import resources
from .data2features import Data2Features

class SmartML():
    def __init__(self, config_file='config.yaml'):
        self.d2f = Data2Features()
        self.time_zone = ZoneInfo("America/New_York")
        self.time_format = '%Y-%m-%d %H:%M:%S'


    def train_val(self, study='ranjan2025', config_file=None, data_file='data_LogReturn_Features_train.txt', metrics_file='metrics.txt', n_splits=10, test_size=250, time_series=True, random_state=42):
        self.load_config(config_file)
        self.df_train = pd.read_table(data_file, header=0, sep='\t')
        self.feature_cols = [x for x in self.df_train.columns if x not in ['datetime', 'class']]
        self.multi_class = (True if len(self.df_train['class'].unique()) > 2 else False)

        if time_series:
            cvs = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        else:
            cvs = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        idx_splits = -1
        for train_idx, val_idx in cvs.split(self.df_train):
            idx_splits += 1
            X_train = self.df_train.loc[train_idx, self.feature_cols] 
            y_train = self.df_train.loc[train_idx, 'class'] 
            X_val = self.df_train.loc[val_idx, self.feature_cols] 
            y_val = self.df_train.loc[val_idx, 'class'] 

            X_train = self.d2f.remove_low_variance(X_train)
            X_train = self.d2f.remove_high_corr(X_train)

            for n_features in self.config['n_features']:
                features = self.d2f.sort_mi_scores(X_train, y_train)['features'][0:n_features]

                X_train = X_train.loc[:, features]
                X_val = X_val.loc[:, features]

                X_train, X_val, scaler = self.d2f.standardize_features(X_train, X_val)

                for model_name, model_info in self.config['models'].items():
                    model_class = eval(model_info['class'])
                    param_grid = model_info['params']
                    keys = list(param_grid.keys())
                    combinations = list(product(*param_grid.values()))
                    for combo in combinations:
                        params = dict(zip(keys, combo))
                        model = self.fit_model(model_name, model_class, params, X_train, y_train, X_val, y_val, random_state)
                        y_pred = model.predict(X_val)
                        y_pred_prob = model.predict_proba(X_val)
                        if self.multi_class:
                            roc_auc = roc_auc_score(y_val, y_pred_prob, multi_class='ovr', average='macro')
                            extra = classification_report(y_val, y_pred, output_dict=True)
                        else:
                            roc_auc = roc_auc_score(y_val, y_pred_prob[:, 1])
                            extra = confusion_matrix(y_val, y_pred).ravel()
                            extra = ','.join([str(x) for x in extra])
                        self.log_metrics([model_name, n_features, str(params), idx_splits] + [roc_auc, 'val', extra], log_file=metrics_file)

    def fit_model(self, study='ranjan2025', model_name, model_class, params, X_train, y_train, X_val=None, y_val=None, random_state=42):
        if model_name == 'mlp':
            model = model_class(**params, random_state=random_state, verbose=0)
            model.fit(X_train, y_train)
        elif model_name == 'catboost':
            model = model_class(**params, random_state=random_state, verbose=0)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), logging_level='Silent')
            else:
                model.fit(X_train, y_train, logging_level='Silent')
        elif model_name == 'lgb':
            model = model_class(**params, random_state=random_state, verbose=-1)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            else:
                model.fit(X_train, y_train)
        elif model_name == 'xgb':
            model = model_class(**params, random_state=random_state)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
        elif model_name == 'random_forest':
            model = model_class(**params, random_state=random_state, verbose=0)
            model.fit(X_train, y_train)
        else:
            model = model_class(**params, random_state=random_state)
            model.fit(X_train, y_train)
        return model

    def eval_model(self, study='ranjan2025', model, X, y, model_name, dataset, log_file):
        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)
        if self.multi_class:
            roc_auc = roc_auc_score(y, y_pred_prob, multi_class='ovr', average='macro')
            extra = classification_report(y, y_pred, output_dict=True)
        else:
            roc_auc = roc_auc_score(y, y_pred_prob[:, 1])
            extra = confusion_matrix(y, y_pred).ravel()
            extra = ','.join([str(x) for x in extra])
        self.log_metrics([model_name] + [roc_auc, dataset, extra], log_file=log_file)

    def get_best_model_params(self, study='ranjan2025', metrics_file='metrics.txt', score='roc_auc', dataset='val'):
        self.metrics = pd.read_table(metrics_file, header=0, sep='\t')

        if self.metrics.shape[0]:
            L = []
            df1 = self.metrics.loc[self.metrics['dataset'] == dataset, ]
            for n_features in df1['n_features'].unique():
                df2 = df1.loc[df1['n_features'] == n_features, ]
                for model in df2['model'].unique():
                    df3 = df2.loc[df2['model'] == model, ]
                    for params in df3['params'].unique():
                        df4 = df3.loc[df3['params'] == params, ]
                        if self.multi_class:
                            extra = '.'
                        else:
                            extra = df4['extra'].str.split(',', expand=True).astype(int).sum(axis=0)
                            extra = ','.join([str(x) for x in extra])
                        sm = np.mean(df4[score])
                        L.append([n_features, model, params] + [sm, dataset, extra])

            ouF = f'{metrics_file.replace(".txt", "_sorted.txt")}'
            ouF2 = f'{metrics_file.replace(".txt", "_sorted_best.txt")}'
            df = pd.DataFrame(L)
            df.columns = ['n_features', 'model', 'params'] + [score, 'dataset', 'extra']
            df.sort_values(by=score, ascending=False, inplace=True)
            df.to_csv(ouF, header=True, index=False, sep='\t')

            n_features = df['n_features'].iloc[0]
            df2 = df.loc[df['n_features'] == n_features, ]
            df3 = df2.drop_duplicates(subset=['model'], keep='first')
            df3.to_csv(ouF2, header=True, index=False, sep='\t')
            self.log('best model params:')
            print(df3)

    def final_fit_eval_train_test(self, study='ranjan2025', data_file='data_LogReturn_Features.txt', config_file='config.yaml', metrics_file='metrics.txt', save_model_file='model.pkl', score='roc_auc', ensemble=True, voting='soft', calibration=False, calibration_cv=5, random_state=42):
        self.load_config(config_file)
        self.df_train = pd.read_table(data_file.replace('.txt', '_train.txt'), header=0, sep='\t')
        self.df_test = pd.read_table(data_file.replace('.txt', '_test.txt'), header=0, sep='\t')
        self.feature_cols = [x for x in self.df_train.columns if x not in ['datetime', 'class']]
        self.multi_class = (True if len(self.df_train['class'].unique()) > 2 else False)
        self.get_best_model_params(score=score, metrics_file=metrics_file)

        to_save = {}
        inF = f'{metrics_file.replace(".txt", "_sorted_best.txt")}'
        df = pd.read_table(inF, header=0, sep='\t')

        X_train = self.df_train.loc[:, self.feature_cols] 
        y_train = self.df_train.loc[:, 'class'] 
        X_test = self.df_test.loc[:, self.feature_cols] 
        y_test = self.df_test.loc[:, 'class'] 


        models = {}
        ouF = f'{metrics_file.replace(".txt", "_eval.txt")}'
        self.log('final eval on full train and test:')
        for n in range(df.shape[0]):
            n_features = df['n_features'].iloc[n]
            params = ast.literal_eval(df['params'].iloc[n])
            model_name = df['model'].iloc[n]
            model_class = eval(self.config['models'][model_name]['class'])

            X_train = self.d2f.remove_low_variance(X_train)
            X_train = self.d2f.remove_high_corr(X_train)
            features = self.d2f.sort_mi_scores(X_train, y_train)['features'][0:n_features]

            X_train = X_train.loc[:, features]
            X_test = X_test.loc[:, features]

            X_train, X_test, scaler = self.d2f.standardize_features(X_train, X_test)

            model = self.fit_model(model_name, model_class, params, X_train, y_train, random_state=random_state)
            models[model_name] = model
            self.eval_model(model, X_train, y_train, model_name, 'train', ouF)
            self.eval_model(model, X_test, y_test, model_name, 'test', ouF)

            to_save['scaler'] = scaler
            to_save['features'] = features
            to_save[model_name] = model

        if ensemble:
            model_name = 'ensemble'
            if calibration:
                for k in models:
                    models[k] = CalibratedClassifierCV(models[k], method='sigmoid', cv=5)

            model = VotingClassifier(estimators=list(models.items()), voting=voting).fit(X_train, y_train)
            self.eval_model(model, X_train, y_train, model_name, 'train', ouF)
            self.eval_model(model, X_test, y_test, model_name, 'test', ouF)
            to_save[model_name] = model

        joblib.dump(to_save, save_model_file)

    def predict(self, study='ranjan2025', model='ensemble', data_file=None, output_file='predicted.txt', data_folder='to_predict', load_model_file='model.pkl',
                symbols='SPY,QQQ,IWM,TLT,^VIX,GLD,CL=F,DX-Y.NYB,EURUSD=X,JPYUSD=X,^TNX,^IRX', target_symbol='SPY',
                start_date=None, end_date=None, market='stock', timeframe='day', interval='1d', period='max', 
                auto_adjust=True, neg_ffill=True, drop_ffill_na=True):

        if data_file is None:
            data_file = f'{data_folder}.txt'
            self.d2f.download_data(symbols=symbols, data_folder=data_folder, data_file=data_file, start_date=start_date, end_date=end_date, interval=interval, market=market, timeframe=timeframe, auto_adjust=auto_adjust, neg_ffill=neg_ffill, period=period)

        self.d2f.get_log_return_labels(data_file=data_file, labeling=False)
        data_file2 = data_file.replace(".txt", "_LogReturn.txt")
        self.d2f.make_features(data_file=data_file2, drop_ffill_na=drop_ffill_na, target_symbol=target_symbol, split=False)

        M = joblib.load(load_model_file)
        features = list(M['features'])
        scaler = M['scaler']
        model = M[model]

        inF = data_file.replace(".txt", "_LogReturn_Features.txt")
        df = pd.read_table(inF, header=0, sep='\t')
        X = df.loc[:, features]
        X, _, _ = self.d2f.standardize_features(X, scaler=scaler)

        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)[:, 1]

        df2 = pd.DataFrame()
        df2['datetime'] = df['datetime']
        df2['pred'] = y_pred
        df2['prob'] = y_pred_prob
        df2.to_csv(output_file, header=True, index=False, sep='\t')

        self.log('final predicted results:')
        print(df2.tail(10))

    def load_config(self, config_file='config.yaml'):
        self.config = {}
        if config_file is not None and os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    self.config = yaml.safe_load(f)  
            except Exception as e:
                self.log(f'loading config {config_file} {e}')
        else:
            config_file = resources.files("smart.config").joinpath('config.yaml')
            if os.path.exists(config_file):
                try:
                    with open(config_file) as f:
                        self.config = yaml.safe_load(f)  
                    self.log('using the default config.yaml')
                except Exception as e:
                    self.log(f'loading the default config {config_file} {e}')
            else:
                self.log('config.yaml is required')

    def log_metrics(self, L, log_file='metrics.txt'):
        current_time = datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        line = '\t'.join([current_time] + [str(x) for x in L])
        self.log(line)

        if log_file.find('eval') == -1:
            header = ['time', 'model', 'n_features', 'params', 'idx_splits'] + ['roc_auc', 'dataset', 'extra']
        else:
            header = ['time', 'model'] + ['roc_auc', 'dataset', 'extra']

        if log_file not in vars(self):
            setattr(self, log_file, True)
            with open(log_file, 'w') as f:
                f.write('\t'.join(header) + '\n')
                f.write(line + '\n')
        else:
            with open(log_file, 'a') as f:
                f.write(line + '\n')

    def log(self, txt=''):
        current_time = datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        print(f'{current_time} {txt}', flush=True)

if __name__ == '__main__':
    sml = SmartML()
    sml.train_val()
    sml.final_fit_eval_train_test()
    sml.predict()

