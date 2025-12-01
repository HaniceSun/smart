import argparse
from .data2features import Data2Features
from .model import SmartML

def parse_args():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser1 = subparsers.add_parser('download', help="download data using yfinance", formatter_class=formatter_class)
    parser1.add_argument('--symbols', type=str, default='SPY,QQQ,IWM,TLT,^VIX,GLD,CL=F,DX-Y.NYB,EURUSD=X,JPYUSD=X,^TNX,^IRX', help="symbols in the cross assets, separated with commas")
    parser1.add_argument('--start_date', type=str, default=None, help="started date of the data (e.g., 2001-01-01), set to the earliest date available if None")
    parser1.add_argument('--end_date', type=str, default=None, help="end date of the data (e.g., 2025-10-31), set to the latest date available if None")
    parser1.add_argument('--data_folder', type=str, default='data', help="folder to save the data")
    parser1.add_argument('--data_file', type=str, default='data.txt', help="output of the merged data")
    parser1.add_argument('--market', type=str, default='stock', help="market of the data")
    parser1.add_argument('--timeframe', type=str, default='day', help="time frame of the data")
    parser1.add_argument('--interval', type=str, default='1d', help="interval of the data")
    parser1.add_argument('--period', type=str, default='max', help="period of the data")
    parser1.add_argument('--auto_adjust', type=bool, default=True, help="back adjust the price using yfinance")
    parser1.add_argument('--neg_ffill', type=bool, default=True, help="forward-fill potential negative price values due to back adjustment if True")

    parser2 = subparsers.add_parser('preprocess', help="calculate log10 return and label each sample based on delta", formatter_class=formatter_class)
    parser2.add_argument('--target_symbol', type=str, default='SPY', help="target symbol to be predicted on, must be in symbols")
    parser2.add_argument('--horizon', type=int, default=5, help="the return of how many days later")
    parser2.add_argument('--delta', type=float, default=-0.01, help="delta to label if sample is a positive or negative case")
    parser2.add_argument('--delta_scale', type=str, default='linear', help="scale of the provided delta (linear or log10), -0.01 means the price dropped 1%% if linear")
    parser2.add_argument('--data_file', type=str, default='data.txt', help="the merged data as input")

    parser3 = subparsers.add_parser('make-features', help="make features from the data", formatter_class=formatter_class)
    parser3.add_argument('--data_file', type=str, default='data_LogReturn.txt', help="the data frame with log10 returns")
    parser3.add_argument('--test_data_start_date', type=str, default='2024-01-01', help="start date of the test dataset used to split the dataset to train and test")
    parser3.add_argument('--target_symbol', type=str, default='SPY', help="target symbol to be predicted on, must be in symbols")
    parser3.add_argument('--drop_ffill_na', type=bool, default=True, help="drop the initial rows of NaN and then forward-fill the rest of the features")

    parser4 = subparsers.add_parser('explore-features', help="explore features about variance, correlation, and mutual information scores on the train dataset", formatter_class=formatter_class)
    parser4.add_argument('--data_file', type=str, default='data_LogReturn_Features_train.txt', help="the train dataset")
    parser4.add_argument('--variance_threshold', type=float, default=1e-4, help="variance threshold")
    parser4.add_argument('--correlation_threshold', type=float, default=0.95, help="Pearson correlation threshold")

    parser5 = subparsers.add_parser('train', help="train the machine learning models configured in configure.yaml", formatter_class=formatter_class)
    parser5.add_argument('--config', type=str, default=None, help="using default config.yaml in the src if not provided")
    parser5.add_argument('--metrics_file', type=str, default='metrics.txt', help="save the metrics during the train to this file")
    parser5.add_argument('--data_file', type=str, default='data_LogReturn_Features_train.txt', help="the train dataset file with features and labels")
    parser5.add_argument('--n_splits', type=int, default=10, help="how many splits")
    parser5.add_argument('--test_size', type=int, default=250, help="the size of each test split")
    parser5.add_argument('--time_series', type=bool, default=True, help="if the data is time series")

    parser6 = subparsers.add_parser('eval', help="evaluate the models on both train and test datasets", formatter_class=formatter_class)
    parser6.add_argument('--data_file', type=str, default='data_LogReturn_Features.txt', help="evaluate on both train and test file")
    parser6.add_argument('--config', type=str, default=None, help="using default config.yaml in the src if not provided")
    parser6.add_argument('--metrics_file', type=str, default='metrics.txt', help="load the metrics from the train")
    parser6.add_argument('--save_model_file', type=str, default='model.pkl', help="save the final model to this file")
    parser6.add_argument('--score', type=str, default='roc_auc', help="rank models by mean of the score on the cross validation folds, pick the top params for each model")
    parser6.add_argument('--ensemble', type=bool, default=True, help="if using ensemble")
    parser6.add_argument('--voting', type=str, default='soft', help="voting methods of ensemble")
    parser6.add_argument('--calibration', type=bool, default=False, help="if doing calibration with ensemble")
    parser6.add_argument('--calibration_cv', type=int, default=5, help="number of CVs if calibration is enabled")

    parser7 = subparsers.add_parser('predict', help="predict new data using the models", formatter_class=formatter_class)
    parser7.add_argument('--model', type=str, default='ensemble', help="using which model to predict")
    parser7.add_argument('--data_file', type=str, default=None, help="download the latest data using yfinance if None")
    parser7.add_argument('--output_file', type=str, default='predicted.txt', help="final prediction results")
    parser7.add_argument('--data_folder', type=str, default='to_predict', help="folder to save the to-predict data")
    parser7.add_argument('--load_model_file', type=str, default='model.pkl', help="the trained model")

    parser7.add_argument('--symbols', type=str, default='SPY,QQQ,IWM,TLT,^VIX,GLD,CL=F,DX-Y.NYB,EURUSD=X,JPYUSD=X,^TNX,^IRX', help="symbols in the cross assets, separated with commas")
    parser7.add_argument('--target_symbol', type=str, default='SPY', help="target symbol to be predicted on, must be in symbols")
    parser7.add_argument('--start_date', type=str, default=None, help="started date of the data (e.g., 2001-01-01), set to the earliest date available if None")
    parser7.add_argument('--end_date', type=str, default=None, help="end date of the data (e.g., 2025-10-31), set to the latest date available if None")
    parser7.add_argument('--market', type=str, default='stock', help="market of the data")
    parser7.add_argument('--timeframe', type=str, default='day', help="time frame of the data")
    parser7.add_argument('--interval', type=str, default='1d', help="interval of the data")
    parser7.add_argument('--period', type=str, default='2y', help="period of the data")
    parser7.add_argument('--auto_adjust', type=bool, default=True, help="back adjust the price using yfinance")
    parser7.add_argument('--neg_ffill', type=bool, default=True, help="forward-fill potential negative price values due to back adjustment if True")
    parser7.add_argument('--drop_ffill_na', type=bool, default=True, help="drop the initial rows of NaN and then forward-fill the rest of the features")

    return parser.parse_args()

def main():
    args = parse_args()

    d2f = Data2Features()
    sml = SmartML()

    if args.command == 'download':
        d2f.download_data(symbols=args.symbols, start_date=args.start_date, end_date=args.end_date, market=args.market, timeframe=args.timeframe, interval=args.interval, period=args.period, neg_ffill=args.neg_ffill)

    if args.command == 'preprocess':
        d2f.get_log_return_labels(target_symbol=args.target_symbol, horizon=args.horizon, delta=args.delta, delta_scale=args.delta_scale, data_file=args.data_file)

    if args.command == 'make-features':
        d2f.make_features(data_file=args.data_file, test_data_start_date=args.test_data_start_date, target_symbol=args.target_symbol, drop_ffill_na=args.drop_ffill_na)

    if args.command == 'explore-features':
        d2f.explore_features(data_file=args.data_file, variance_threshold=args.variance_threshold, correlation_threshold=args.correlation_threshold)

    if args.command == 'train':
        sml.train_val(config_file=args.config, data_file=args.data_file, metrics_file=args.metrics_file, n_splits=args.n_splits, test_size=args.test_size, time_series=args.time_series)

    if args.command == 'eval':
        sml.final_fit_eval_train_test(data_file=args.data_file, metrics_file=args.metrics_file, score=args.score, ensemble=args.ensemble, voting=args.voting, calibration=args.calibration, calibration_cv=args.calibration_cv)

    if args.command == 'predict':
        sml.predict(model=args.model, data_file=args.data_file, output_file=args.output_file, data_folder=args.data_folder, load_model_file=args.load_model_file, symbols=args.symbols, target_symbol=args.target_symbol, start_date=args.start_date, end_date=args.end_date, market=args.market, timeframe=args.timeframe, interval=args.interval, period=args.period, auto_adjust=args.auto_adjust, neg_ffill=args.neg_ffill)

if __name__ == '__main__':
    main()
