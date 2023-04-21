from expert.src.evaluator import Evaluator
import pandas as pd
import os
from expert.src.utils import get_dmax
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import trange
import numpy as np
from joblib import Parallel

class args(object):
    def __init__(self):
        self.model = 'test/independent_model'
        self.input = 'test/independent_results'
        self.output = 'test/independent_eval'
        self.labels = 'test/y_test.h5'
        self.silence = False
        self.in_cm = True
        self.db_file = '/home/zhanghaohong/.etetoolkit/taxa.sqlite'
        self.otlg = 'test/ontology.pkl'
        self.measure_unknown = False
        self.processors = -1
        self.T = 100
        self.S = -1
        self.dropout_rate = 0.1
        self.val_split = 0.1
        self.hide_warnings = True
        self.rg = True
        
def R2(y_true, y_pred):
    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()
    cov = np.cov(y_true, y_pred, rowvar=False)
    return cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])

def evaluate(cfg, args):
    if args.rg:
        predictions = pd.read_csv(os.path.join(args.input, 'predicted.csv'), index_col=0)
        sources = pd.read_hdf(args.labels, key='rg').loc[predictions.index, :]
        mae, mse, r2 = mean_absolute_error(sources, predictions), mean_squared_error(sources, predictions), R2(sources, predictions)
        result_df = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'R2': [r2]})
        os.makedirs(args.output, exist_ok=True)
        result_df.to_csv(os.path.join(args.output, 'result.csv'))
        return 
        
    layers = [os.path.join(args.input, i) for i in sorted(os.listdir(args.input), key=lambda x: int(x.split('.')[0].split('-')[1]))]
    np.random.seed(0)
    #idx = np.random.choice(np.arange(100000), 10000)
    predictions = [pd.read_csv(layer, index_col=0)#.iloc[idx, :]
                   for layer in layers]
    sources = [pd.read_hdf(args.labels, key='l'+str(layer))#.iloc[idx, :]
               for layer in range(get_dmax(args.labels))]

    if 'root' in sources[0].columns:
        sources = sources[1:]
        contains_root = 1

    print('Reordering labels and prediction result')
    IDs = list(set(predictions[0].index.to_list()).intersection(sources[0].index.to_list()))

    sources = [source_singlelayer.loc[IDs, :] for source_singlelayer in sources]
    predictions = [predictions_singlelayer.loc[IDs, :] for predictions_singlelayer in predictions]
    print('Reordering labels and prediction result for samples')

    par = Parallel(n_jobs=args.processors, backend='loky')
    print('Running evaluation...')
    evaltr = Evaluator(predictions_multilayer=predictions, actual_sources_multilayer=sources,
                       num_thresholds=args.T, sample_count_threshold=args.S, par=par)
    metrics_layers, avg_metrics_layers, overall_metrics = evaltr.eval()
    print('Saving evaluation results...')
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    overall_metrics.to_csv(os.path.join(args.output, 'overall.csv')) # debug
    for layer in trange(len(layers)):
        if not os.path.isdir(os.path.join(args.output, 'layer-' + str(layer+2))):
            os.mkdir(os.path.join(args.output, 'layer-' + str(layer+2)))
        metrics_layer = metrics_layers[layer]
        avg_metrics_layer = avg_metrics_layers[layer]
        avg_metrics_layer.to_csv(os.path.join(args.output, 'layer-' + str(layer+2) + '.csv' ))
        for label, metrics in metrics_layer.items():
            metrics.to_csv(os.path.join(args.output, 'layer-' + str(layer+2), label + '.csv'))

    '''
    pd.concat(map(lambda x: pd.read_csv(x, index_col=0).loc[0.00, ['ROC-AUC', 'F-max']].rename(os.path.split(x)[1].split('.')[0]), all_metrics), axis=1).T
    paths = ['EvalResult_{}'.format(i) for i in range(5)]
    metrics = [path2metrics(path) for path in paths]
    metrics = pd.concat(metrics, axis=1)
    metrics.T.groupby(by=metrics.T.index).mean().T.rename(columns=lambda x: '(Avg.) '+x).round(4)
    '''

