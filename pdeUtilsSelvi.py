# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:30:57 2022
Set of Utility Functions
@author: Taufik Sutanto
"""
import warnings; warnings.simplefilter('ignore')
import darts, pickle, pandas as pd, nolds, numpy as np, os, sys
import matplotlib.pyplot as plt, seaborn as sns, psutil, csv
import plotly.graph_objects as go, scipy.io
#import pdeDatasets as pde, torch #, kats, 
# import torch
#from kats.tsfeatures.tsfeatures import TsFeatures
from darts.models import KalmanFilter, GaussianProcessFilter#, MovingAverage
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from darts import metrics, models
from plotly.subplots import make_subplots

plt.style.use('bmh'); sns.set()
SEED=0; np.random.seed(SEED) # Set seed for the Random Number Generator (RNG)
#dtype = torch.float
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def backTest(seriesCV, modelFile_, noise_=False, method_='',  DL=False, dist_=0.0, start_=0, horizon_=30, metric_='smape', kCV=10):
    mainMetric = getattr(metrics, metric_)
    resCV = []
    for i in range(kCV):
        s_, e_ = start_+i*horizon_, start_+(i+1)*horizon_
        trainData = seriesCV.slice(0, s_) # Expanding Windows
        if noise_ and dist_:
            trainData = addNoise(trainData, dist_=dist_)
        valData = seriesCV.slice(s_, e_)
        if DL:
            if method_.lower().strip() in {'rnn', 'lstm', 'gru', 'RNNModel'}:
                alg_ = getattr(models, 'RNNModel')
            else:
                alg_ = getattr(models, method_)
            model = alg_.load_model((modelFile_.replace(".pckl", ".pth.tar")))
        else:
            model, _, _ = load(modelFile_, DL=False) # Best Parameter from Hyperparameter Tunning
        model.fit(trainData)
        forecast_ = model.predict(len(valData))
        resCV.append(mainMetric(valData, forecast_))
        del model
    return resCV

def deNoising(yTrain=None, yForecast=None, alg_='kalman', dim_x=1, alpha=0.1, window=7):
    if alg_.lower().strip() in {'kalman', 'kalmanfilter'}:
        kf = KalmanFilter(dim_x=dim_x)
        kf.fit(yForecast, yTrain)
        return kf.filter(yForecast, yTrain)
    elif alg_.lower().strip() in {'gaussian', 'gaussianprocessfilter'}:
        gpf = GaussianProcessFilter(kernel=RBF(), alpha=alpha, normalize_y=True)
        return gpf.filter(yForecast, num_samples=30)
    elif alg_.lower().strip() in {'ma', 'movingaverage'}:
        pass
        #return MovingAverage(window=window).filter(yForecast)
    else:
        print("Unsupported denoising algorithm, returning origin TimeSeries")
        return yForecast

def describe(data, basicStat_=True, verbose=True, visual_=True, sample_=0, colSample=100, start_=0, Name='', noXthick=False, figSize=(12,6)):
    dd  = {}
    if data.shape[1]>colSample:
        data = data.iloc[:,-colSample:]
    if basicStat_:
        dd = data.describe() # print(dd.columns)
        dd.rename({'50%': 'median'}, axis=0, inplace=True)
        dd.drop(['min','max','count','25%', '75%'], axis=0, inplace=True)
        dd.loc['Entropy'] = [nolds.sampen(data[col_]) for col_ in data.columns] # Sample Entropy
        dd.loc['Lyapunov Exp.'] = [nolds.lyap_r(data[col_]) for col_ in data.columns] #lyap_r
        dd.loc['Hurst Exp.'] = [nolds.hurst_rs(data[col_]) for col_ in data.columns]
        dd.loc['DFA'] = [nolds.dfa(data[col_]) for col_ in data.columns] #Detrended Fluctuation Analysis
        if dd.shape[1]>1:
            if str(dd.columns[1])!=str(dd.columns[0])+'h':
                if Name: dd[Name] = dd.mean(numeric_only=True, axis=1)
                else: dd[''.join(dd.columns)] = dd.mean(numeric_only=True, axis=1)
        elif Name: dd.rename(columns={dd.columns[0]: Name}, inplace=True)
        if verbose:
            if len(dd.columns)<colSample: print("Descriptive & Dynamical Measures = '{}' \n".format(Name), dd)
            else: print("Descriptive & Dynamical Measures ({} last columns) of '{}' = \n".format(colSample, Name), dd.iloc[: , -4:])
    if visual_:
        plt.figure(figsize=figSize)
        if sample_>0:
            p = sns.lineplot(data=data[start_:start_+sample_], palette='Set1')
            if noXthick:
                p.set(xticklabels=[])
                p.set(xlabel=None)
                p.tick_params(bottom=False)  # remove the ticks
        else: _ = sns.lineplot(data=data, palette='Set1')
        plt.show()
    return dd

def dfStandardize(df, type_='minMax'):
    if type_.lower().strip() == 'minmax':
        dfTmp = MinMaxScaler().fit_transform(df)
    else:
        dfTmp = StandardScaler().fit_transform(df)
    return pd.DataFrame(dfTmp, index=df.index, columns=df.columns)

def plotKSE():
    try:
        dt = scipy.io.loadmat('data/KSE.mat')
    except:
        dt = scipy.io.loadmat('../data/KSE.mat')
    u = dt['U']#.flatten()
    x, tt = np.meshgrid(dt["x"], dt['tt'])
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
    fig.add_trace(go.Surface(x=tt, y=x, z=u, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.update_layout(title_text='Kuramoto-Sivashinsky', height=400, width=800)
    fig.update_layout(scene=dict(xaxis_title='t', yaxis_title='x', zaxis_title='U'))
    fig.show()
    return True

def plotWave():
    try:
        dt = scipy.io.loadmat('data/1Dwave.mat')
    except:
        dt = scipy.io.loadmat('../data/1Dwave.mat')
    u = dt['U']#.flatten()
    x, t = np.meshgrid(dt["x"], dt['t'])
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
    fig.add_trace(go.Surface(x=t, y=x, z=u, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.update_layout(title_text='1D Wave System', height=400, width=800)
    fig.update_layout(scene=dict(xaxis_title='t', yaxis_title='x', zaxis_title='U'))
    fig.show()
    return True

"""
def plotNLS():
    _, dfr, dfi = pde.dataset('NLS', verbose=True, standardize='minMax', file_='../data/NLS/train_AB3.csv', plot_=False)
    r, c = dfr.shape
    t = np.linspace(0, 105, num=r)
    x = np.linspace(-7.8935, 7.831880, num=c)
    X, T = np.meshgrid(x, t)
    Exact_u = dfr.to_numpy()
    Exact_v = dfi.to_numpy()
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
    fig.add_trace(go.Surface(x=X, y=T, z=Exact_u, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=X, y=T, z=Exact_v, colorscale='RdBu', showscale=False), row=1, col=2)
    fig.update_layout(title_text='Non-Linear Schrodinger Golden Standard', height=400, width=900)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='T', zaxis_title='U (real)'),
                      scene2=dict(xaxis_title='X', yaxis_title='T', zaxis_title='V (imajiner)'),)
    fig.show()
    return True
"""

def HistForecast(model, series, start_=0.5, horizon_=100, stride_=100, metrics='rmse'):
    backtest = model.historical_forecasts(series=series,start=start_, forecast_horizon=horizon_, stride=stride_)
    if metrics.lower().strip()=='rmse': error = darts.metrics.rmse(backtest, series)
    elif metrics.lower().strip()=='mape': error = darts.metrics.mape(backtest, series)
    return error

def addNoise(series, dist_=0.01):
    tg = darts.utils.timeseries_generation
    size_ = len(series)
    noisySeries = []
    for col in series.columns:
        noise_ = tg.gaussian_timeseries(length=size_, std=dist_)
        noisySeries.append(series[col]+noise_)

    for i, s in enumerate(noisySeries):
        if i==0:
            Result = s
        else:
            Result = Result.stack(s)
    return Result

def df2series(data, pTrain=0.5, pTest=0.2, prec_=32, merge=False, noise_=False, dist_=0.05):
    if pTest+pTrain>1.0:
        sys.exit("Error pTest+pTrain>1 Exitting...")
    d = data.head(int(data.shape[0]*(pTrain+pTest)))
    if merge:
        for i, col in enumerate(data.columns):
            if i==0:
                series = darts.TimeSeries.from_dataframe(d, time_col=None, value_cols=col)
            else:
                series = series.stack(darts.TimeSeries.from_dataframe(d, time_col=None, value_cols=col))
        if noise_:
            series = addNoise(series, dist_=dist_)
        if prec_==32:
            series = series.astype(np.float32)
        return series, None
    else:
        dTrain = d.head(int(data.shape[0]*(pTrain)))
        dTest = d.tail(int(data.shape[0]*(pTest)))
        for i, col in enumerate(data.columns):
            if i==0:
                train = darts.TimeSeries.from_dataframe(dTrain, time_col=None, value_cols=col)
                test = darts.TimeSeries.from_dataframe(dTest, time_col=None, value_cols=col)
            else:
                train = train.stack(darts.TimeSeries.from_dataframe(dTrain, time_col=None, value_cols=col))
                test = test.stack(darts.TimeSeries.from_dataframe(dTest, time_col=None, value_cols=col))
        if noise_:
            train = addNoise(train, dist_=dist_)
            test = addNoise(test, dist_=dist_)
        if prec_==32:
            train = train.astype(np.float32)
            test = test.astype(np.float32)
        return train, test

def save(modelFile_, model):
    with open(modelFile_, "wb") as handle:
        pickle.dump(model, handle)

def load(modelFile_):
    with open(modelFile_, "rb") as input_file:
        return pickle.load(input_file)

def memory_usage():
    # return the memory usage in bytes
    # Memory usage calculated as Max memory in the iterations & in pyshical memory (RSS):
    # https://stackoverflow.com/questions/938733/total-memory-used-by-python-process/21632554#21632554
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def RemCSVblanks(file_):
    with open(file_) as in_file:
        with open(file_.replace(".csv","_cleaned.csv"), 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            for row in csv.reader(in_file):
                if row or len(row)>1:
                    writer.writerow(row)
    os.remove(file_)
    os.rename(file_.replace(".csv","_cleaned.csv"), file_)

def distMetric(mInput, mTarget):
    dist_ = 0.0
    for k, v in mInput.items():
        dist_ += (v-mTarget[k])**2
    return dist_/len(mInput)

"""
def fMetrics(seq, dict_=False, ts_=False, nl_=True):
    if ts_:
        katsModel = TsFeatures()
        TS_features = katsModel.transform(pd.DataFrame(data=seq, columns=['value']))
        features = 'var lumpiness stability heterogeneity linearity trend_strength seasonality_strength spikiness'.split()

    if dict_:
        metric_ = {}
        if nl_:
            metric_['Entropy'] = nolds.sampen(seq) # Sample Entropy
            metric_['Lyapunov Exp.'] = nolds.lyap_r(seq) #lyap_r
            metric_['Hurst Exp.'] = nolds.hurst_rs(seq)
            metric_['DFA'] = nolds.dfa(seq) #Detrended Fluctuation Analysis
        if ts_:
            for feature in features:
                metric_[feature] = TS_features[feature]
    else:
        metric_ = []
        if nl_:
            metric_.append(nolds.sampen(seq))
            metric_.append(nolds.lyap_r(seq))
            metric_.append(nolds.hurst_rs(seq))
            metric_.append(nolds.dfa(seq))
        if ts_:
            for feature in features:
                metric_.append(TS_features[feature])
        metric_ = np.array(metric_)
    return metric_

class PDELossNL(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PDELossNL, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.mean((targets - inputs)**2)

        mInput = torch.reshape(inputs, (-1,)).cpu().detach().numpy().squeeze()
        mInput = fMetrics(mInput, ts_=False, nl_=True)
        mTarget = torch.reshape(targets, (-1,)).cpu().detach().numpy().squeeze()
        mTarget = fMetrics(mTarget, ts_=False, nl_=True)
        metric_ = np.mean((mInput - mTarget)**2)

        return loss + metric_

class PDELossTS(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PDELossTS, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.mean((targets - inputs)**2)

        mInput = torch.reshape(inputs, (-1,)).cpu().detach().numpy().squeeze()
        mInput = fMetrics(mInput, ts_=True, nl_=False)
        mTarget = torch.reshape(targets, (-1,)).cpu().detach().numpy().squeeze()
        mTarget = fMetrics(mTarget, ts_=True, nl_=False)
        metric_ = np.mean((mInput - mTarget)**2)

        return loss + metric_

class PDELossTSNL(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PDELossTSNL, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.mean((targets - inputs)**2)

        mInput = torch.reshape(inputs, (-1,)).cpu().detach().numpy().squeeze()
        mInput = fMetrics(mInput, ts_=True, nl_=True)
        mTarget = torch.reshape(targets, (-1,)).cpu().detach().numpy().squeeze()
        mTarget = fMetrics(mTarget, ts_=True, nl_=True)
        metric_ = np.mean((mInput - mTarget)**2)

        return loss + metric_

"""
import stumpy

def subMatch(trainSeq, sampleSeq, Lout=100):
    train = trainSeq.values()[:,0].astype(np.float64) # Get only first column
    seq = sampleSeq.values()[:,0].astype(np.float64)
    distance_profile = stumpy.mass(seq, train)
    idx = np.argmin(distance_profile)
    #print(f"The nearest neighbor to is located at index {idx}")
    start_ = idx+len(seq)
    end_ = start_ + Lout
    subSeq = train[start_ : end_]
    return subSeq, idx

def checkSubMatch(trainSeq, sampleSeq, idx):
    train = trainSeq.values()[:,0].astype(np.float64) # Get only first column
    seq = sampleSeq.values()[:,0].astype(np.float64)
    Q_z_norm = stumpy.core.z_norm(seq)
    nn_z_norm = stumpy.core.z_norm(train[idx:idx+len(seq)])
    plt.suptitle('Comparing The Query To Its Nearest Neighbor', fontsize='14')
    plt.xlabel('Time', fontsize ='12')
    plt.ylabel('Time Series Sequence', fontsize='12')
    plt.plot(Q_z_norm, lw=2, color="C1", label="Subsequence")
    plt.plot(nn_z_norm, lw=2, label="Nearest Subsequence")
    plt.legend()
    plt.show()
