"""
Linear models to forecast the realized correlations, including HAR and GHAR. HAR is a special case of GHAR, assuming the adjacency matrix is identity.
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=22, help="forward-looking period")
parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon")
parser.add_argument("--model_name", type=str, default='GHAR', help="model name")
parser.add_argument("--adj_name", type=str, default='iden+line', help="model name")
parser.add_argument("--universe", type=str, default='DJIA', help="data name")
parser.add_argument("--version", type=str, default='Forecast_Corr', help="version name")

opt = parser.parse_args()
print(opt)

# Specific version
this_version = '_'.join(
    [opt.version,
     opt.model_name,
     opt.adj_name,
     opt.universe,
     'W' + str(opt.window),
     'F' + str(opt.horizon)])


path = 'your_local_path'
model_save_path = join('your_model_storage_path', this_version)
os.makedirs(model_save_path, exist_ok=True)


def load_feature_data(universe):
    feature_df = pd.read_csv(join(path, 'Data', f'{universe}_corr_FH1.csv'), index_col=0)
    feature_df.fillna(method="ffill", inplace=True)
    feature_df = feature_df[feature_df.index <= '2021-07-01']
    feature_df = feature_df.sort_index(axis=1)
    return feature_df


def load_data(universe, horizon):
    corr_df = pd.read_csv(join(path, 'Data', f'{universe}_corr_FH{horizon}.csv'), index_col=0)
    corr_df.fillna(method="ffill", inplace=True)
    vech_df = corr_df[corr_df.index <= '2021-07-01']
    vech_df = vech_df.sort_index(axis=1)
    return vech_df


def preprocess_HAR(feature_df, vech_df):
    subdf_l = []
    all_assets_l = [i for i in vech_df.columns if i not in ['Date', 'Time']]
    all_assets_l.sort()

    har_lags = [1, 5, 22]

    for target_corr in vech_df:
        subdf = pd.DataFrame()
        subdf['Target'] = vech_df[target_corr].copy()
        subdf['Date'] = vech_df.index
        subdf['Ticker'] = target_corr
        indpt_df_l = []
        for lag in har_lags:
            tmp_indpdt_df = 0
            for il in range(1, 1+lag):
                tmp_indpdt_df += feature_df[target_corr].shift(il)

            indpt_df_l.append(tmp_indpdt_df / lag)

        # reverse the time order
        explain_df = pd.concat(indpt_df_l, axis=1)
        explain_df.columns = ['corr+lag%d' % i for i in har_lags]

        subdf = pd.merge(subdf, explain_df, left_index=True, right_index=True)
        subdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        subdf.dropna(inplace=True)
        subdf_l.append(subdf)

    df = pd.concat(subdf_l)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_HAR_Global(feature_df, vech_df):
    subdf_l = []
    all_assets_l = [i for i in vech_df.columns if i not in ['Date', 'Time']]
    all_assets_l.sort()

    har_lags = [1, 5, 22]

    global_df = feature_df[all_assets_l].mean(1)

    for target_corr in vech_df:
        subdf = pd.DataFrame()
        subdf['Target'] = vech_df[target_corr].copy()
        subdf['Date'] = vech_df.index
        subdf['Ticker'] = target_corr
        indpt_df_l = []
        glb_df_l = []
        for lag in har_lags:
            tmp_indpdt_df = 0
            tmp_glb_df = 0
            for il in range(1, 1+lag):
                tmp_indpdt_df += feature_df[target_corr].shift(il)
                tmp_glb_df += global_df.shift(il)

            indpt_df_l.append(tmp_indpdt_df / lag)
            glb_df_l.append(tmp_glb_df / lag)

        # reverse the time order
        explain_df = pd.concat(indpt_df_l+glb_df_l, axis=1)
        explain_df.columns = ['corr+lag%d' % i for i in har_lags] + ['glbcorr+lag%d' % i for i in har_lags]

        subdf = pd.merge(subdf, explain_df, left_index=True, right_index=True)
        subdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        subdf.dropna(inplace=True)
        subdf_l.append(subdf)

    df = pd.concat(subdf_l)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_HAR_Line(feature_df, vech_df):
    subdf_l = []
    all_assets_l = [i for i in vech_df.columns if i not in ['Date', 'Time']]
    all_assets_l.sort()

    har_lags = [1, 5, 22]

    for target_corr in vech_df:
        s0 = target_corr.split('-')[0]
        s1 = target_corr.split('-')[1]
        connect_pairs = [i for i in all_assets_l if s0+'-' in i or '-'+s0 in i or s1+'-' in i or '-'+s1 in i]
        subdf = pd.DataFrame()
        subdf['Target'] = vech_df[target_corr].copy()
        subdf['Date'] = vech_df.index
        subdf['Ticker'] = target_corr
        indpt_df_l = []
        glb_df_l = []
        for lag in har_lags:
            tmp_indpdt_df = 0
            tmp_glb_df = 0
            for il in range(1, 1+lag):
                tmp_indpdt_df += feature_df[target_corr].shift(il)
                tmp_glb_df += feature_df[connect_pairs].mean(1).shift(il)

            indpt_df_l.append(tmp_indpdt_df / lag)
            glb_df_l.append(tmp_glb_df / lag)

        # reverse the time order
        explain_df = pd.concat(indpt_df_l+glb_df_l, axis=1)
        explain_df.columns = ['corr+lag%d' % i for i in har_lags] + ['linecorr+lag%d' % i for i in har_lags]

        subdf = pd.merge(subdf, explain_df, left_index=True, right_index=True)
        subdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        subdf.dropna(inplace=True)
        subdf_l.append(subdf)

    df = pd.concat(subdf_l)
    df.reset_index(drop=True, inplace=True)
    return df


def df2arr(df, corrs_l):
    all_inputs = df[corrs_l].values
    all_targets = df[['Target']].values
    return all_inputs, all_targets


def Train(df, date, date_l):
    timestamp = date_l.index(date)
    # split time
    s_p = max(timestamp-1000, 0)
    f_p = min(timestamp + opt.window, len(date_l)-1)

    s_date = date_l[s_p]
    f_date = date_l[f_p]

    corrs_l = [i for i in df.columns if 'lag' in i]
    # split data
    train_df = df[df['Date'] >= s_date]
    train_df = train_df[train_df['Date'] < date]

    test_df = df[df['Date'] >= date]
    test_df = test_df[test_df['Date'] < f_date]

    train_x, train_y = df2arr(train_df, corrs_l)
    test_x, test_y = df2arr(test_df, corrs_l)
    print(test_x.shape)

    # Augmented
    best_model = LinearRegression()
    best_model.fit(train_x, train_y)

    pred_y = best_model.predict(test_x)
    print('Before: [%.3f, %.3f]' % (pred_y.min(), pred_y.max()))
    pred_y[pred_y<=-1] = -1
    pred_y[pred_y>1] = 1
    print('After: [%.3f, %.3f]' % (pred_y.min(), pred_y.max()))

    test_pred_df = test_df[['Ticker', 'Date']]
    test_pred_df['Pred_VHAR'] = pred_y
    test_pred_df = test_pred_df.pivot(index='Date', columns='Ticker', values='Pred_VHAR')

    test_pred_df.columns = list(test_pred_df.columns)
    test_pred_df.index = list(test_pred_df.index)

    save_path = join(path, 'Corr_Pred_Results', this_version)
    os.makedirs(save_path, exist_ok=True)

    test_pred_df.to_csv(join(save_path, 'Pred_%s.csv' % date))


def connect_pred():
    save_path = join(path, 'Corr_Pred_Results', this_version)
    files_l = os.listdir(save_path)
    pred_files = [i for i in files_l if 'Pred_' in i]
    pred_files.sort()
    test_pred_df_l = []
    for i in pred_files:
        test_pred_df = pd.read_csv(join(save_path, i), index_col=0)
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l)
    print(test_pred_df)

    sum_path = join(path, 'Corr_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    feature_df = load_feature_data(opt.universe)
    vech_df = load_data(opt.universe, opt.horizon)
    print(vech_df)

    if opt.adj_name == 'iden+global':
        df = preprocess_HAR_Global(feature_df, vech_df)
    elif opt.adj_name == 'iden+line':
        df = preprocess_HAR_Line(feature_df, vech_df)
    else:
        df = preprocess_HAR(feature_df, vech_df)

    print(df.shape)
    date_l = list(set(df['Date'].tolist()))
    date_l.sort()

    print('Training Starts Now ...')
    idx = date_l.index('2011-07-01')
    for date in date_l[idx::opt.window]:
        print(' * ' * 20 + date + ' * ' * 20)
        Train(df, date, date_l)

    connect_pred()