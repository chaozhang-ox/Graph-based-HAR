"""
Combine the forecasted variance and correlation to get the forecasted covariance matrix
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

path = 'your_local_path'
model_save_path = join('your_model_storage_path', this_version)
os.makedirs(model_save_path, exist_ok=True)


var_parser = argparse.ArgumentParser()
var_parser.add_argument("--window", type=int, default=22, help="forward-looking period")
var_parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon")
var_parser.add_argument("--model_name", type=str, default='GHAR', help="model name")
var_parser.add_argument("--adj_name", type=str, default='iden+glasso', help="model name")
var_parser.add_argument("--universe", type=str, default='DJIA', help="data name")
var_parser.add_argument("--version", type=str, default='Forecast_Var', help="version name")

var_opt = var_parser.parse_args()
print(var_opt)

var_version = '_'.join(
    [var_opt.version,
     var_opt.model_name,
     var_opt.adj_name,
     var_opt.universe,
     'W' + str(var_opt.window),
     'F' + str(var_opt.horizon)])

corr_parser = argparse.ArgumentParser()
corr_parser.add_argument("--window", type=int, default=22, help="forward-looking period")
corr_parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon")
corr_parser.add_argument("--model_name", type=str, default='GHAR', help="model name")
corr_parser.add_argument("--adj_name", type=str, default='gamma', help="model name")
corr_parser.add_argument("--universe", type=str, default='DJIA', help="data name")
corr_parser.add_argument("--version", type=str, default='Forecast_Corr', help="version name")

corr_opt = corr_parser.parse_args()
print(corr_opt)

corr_version = '_'.join(
    [corr_opt.version,
     corr_opt.model_name,
     corr_opt.adj_name,
     corr_opt.universe,
     'W' + str(corr_opt.window),
     'F' + str(corr_opt.horizon)])

this_version = '_'.join(
    ['Forecast_Cov',
     var_opt.model_name + '+' + var_opt.adj_name +'-' + corr_opt.model_name + '+' + corr_opt.adj_name,
     corr_opt.universe,
     'W' + str(corr_opt.window),
     'F' + str(corr_opt.horizon)])


def load_var(universe, horizon):
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)
    vech_df = var_df[var_df.index <= '2021-07-01']
    vech_df = vech_df.sort_index(axis=1)
    return vech_df


def load_corr(universe, horizon):
    corr_df = pd.read_csv(join(path, 'Data', f'{universe}_corr_FH{horizon}.csv'), index_col=0)
    corr_df.fillna(method="ffill", inplace=True)
    vech_df = corr_df[corr_df.index <= '2021-07-01']
    vech_df = vech_df.sort_index(axis=1)
    return vech_df


def vec2matrix(vech):
    n = int(np.sqrt(2*len(vech)+0.25) + 0.5)
    inv_S = np.ones((n, n))
    for i in range(n):
        tmp_n = int((i-1)*i/2)
        inv_S[i, :i] = vech[tmp_n:tmp_n+i]
        inv_S[:i, i] = vech[tmp_n:tmp_n+i]

    return inv_S


def inv_DRD(var_vec, corr_vec):
    corr_m = vec2matrix(corr_vec)
    vol_m = np.diag(np.sqrt(var_vec))

    cov_m = np.matmul(vol_m, corr_m)
    cov_m = np.matmul(cov_m, vol_m)
    return cov_m


def cholesky_decompose(A):
    L = np.linalg.cholesky(A)
    n = A.shape[0]
    vech = []
    for i in range(n):
        vech.extend(L[i, :i+1])

    vech = np.array(vech)
    return vech


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def connect_pred(var_df, corr_df):
    var_sum_path = join(path, 'Var_Results_Sum')
    var_pred_df = pd.read_csv(join(var_sum_path, var_version + '_pred.csv'), index_col=0)
    var_pred_df = var_pred_df[var_df.columns]
    corr_sum_path = join(path, 'Corr_Results_Sum')
    corr_pred_df = pd.read_csv(join(corr_sum_path, corr_version + '_pred.csv'), index_col=0)
    corr_pred_df = corr_pred_df[corr_df.columns]

    date_l = corr_pred_df.index.to_list()
    chol_vech_l = []

    for i, date in enumerate(date_l):
        var_vec = var_pred_df.loc[date]
        corr_vec = corr_pred_df.loc[date]

        cov_m = inv_DRD(var_vec.values, corr_vec.values)

        if not is_pos_def(cov_m):
            print(date)

        try:
            chol_vech = cholesky_decompose(cov_m)
        except:
            print(date)
            var_vec = var_df.loc[date_l[i-1]]
            corr_vec = corr_df.loc[date_l[i-1]]
            cov_m = inv_DRD(var_vec.values, corr_vec.values)
            chol_vech = cholesky_decompose(cov_m)

        chol_vech_l.append(chol_vech)

    chol_vech_df = pd.DataFrame(chol_vech_l, index=date_l)
    chol_vech_df.columns = ['v'+str(i).zfill(4) for i in range(chol_vech_df.shape[1])]

    print(chol_vech_df)
    sum_path = join(path, 'Cov_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    chol_vech_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    var_df = load_var(var_opt.universe, var_opt.horizon)
    corr_df = load_corr(corr_opt.universe, corr_opt.horizon)
    connect_pred(var_df, corr_df)