"""
Use forecasted covariance matrix to compute the GMVP+ portfolio
Evaluate the performance of the GMVP+ portfolio
"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import os
from os.path import *
import pandas as pd
import numpy as np
from MCS import *


path = 'your_local_path'
model_save_path = join('your_model_storage_path', this_version)
os.makedirs(model_save_path, exist_ok=True)

sum_path = join(path, 'Cov_Results_Sum')


def load_ret(universe):
    ret_df = pd.read_csv(join(path, 'Data', f'{universe}_ret_FH1.csv'), index_col=0)
    ret_df.fillna(method="ffill", inplace=True)
    ret_df = ret_df[ret_df.index <= '2021-07-01']
    ret_df = ret_df.sort_index(axis=1)
    return ret_df


def load_data(universe, horizon):
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_Chol_FH1.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)
    vech_df = var_df[var_df.index <= '2021-07-01']
    return vech_df


def square_cholesky(vech):
    n = int(np.sqrt(2*len(vech)+0.25) - 0.5)
    L_inv = np.zeros((n, n))
    for i in range(n):
        tmp_n = int((i+1)*i/2)
        L_inv[i, :i+1] = vech[tmp_n:tmp_n+i+1]

    inv_S = np.dot(L_inv, L_inv.T)
    return inv_S


def GMVPPlus(H):
    import cvxpy as cp
    n = H.shape[0]
    x = cp.Variable(n)
    A = np.eye(n)
    q = np.ones(n)

    prob = cp.Problem(cp.Minimize(cp.quad_form(x, H)),
                      [q.T @ x == 1,
                       (-A) @ x <= 0])
    prob.solve()

    x.value[np.abs(x.value) < 1e-6] = 0
    w_value = np.round(x.value, 6)
    # print("A solution by CVXPY is")
    # print(w_value)
    return w_value


def Turnover(w, prev_w, ret):
    new_w = prev_w * (1+ret) / (1 + np.dot(prev_w, ret))
    return np.sum(np.abs(w - new_w))


def square_lower(vech):
    n = int(np.sqrt(2*len(vech)+0.25) - 0.5)
    inv_S = np.zeros((n, n))
    for i in range(n):
        tmp_n = int((i+1)*i/2)
        inv_S[i, :i+1] = vech[tmp_n:tmp_n+i+1]
        inv_S[:i+1, i] = vech[tmp_n:tmp_n+i+1]

    return inv_S


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def Loss(ret_df, vech_df, test_pred_df, port, idx, horizon):
    date_l = list(test_pred_df.index)[idx::horizon]
    # print(date_l)
    test_vech_df = vech_df.loc[date_l]
    test_ret_df = ret_df.loc[date_l] / 1e2
    r_p_l = []
    to_p_l = []
    dist_l = []

    prev_w = np.zeros(test_ret_df.shape[1])

    for i, date in enumerate(date_l):
        if i % 500 == 0:
            print(date)
        ret_vech = test_ret_df.loc[date]
        if port == '1/N':
            pred_vech = np.zeros(len(ret_vech))
        elif port == 'RC':
            pred_vech = test_vech_df.loc[date]
        else:
            pred_vech = test_pred_df.loc[date]

        H_t = square_cholesky(pred_vech)
        S_t = square_cholesky(test_vech_df.loc[date])

        if is_pos_def(H_t):
            pass
        else:
            pre_date = date_l[date_l.index(date) - 1]
            pre_vech = test_vech_df.loc[pre_date]
            H_t = square_cholesky(pre_vech)

        if port == '1/N':
            w = np.ones_like(ret_vech) / len(ret_vech)
        else:
            w = GMVPPlus(H_t)

        w_R = GMVPPlus(S_t)

        r_p = np.dot(ret_vech, w)

        r_p_l.append(r_p)

        to_p_l.append(Turnover(w, prev_w, ret_vech))
        dist_l.append(np.linalg.norm(w-w_R))

        prev_w = w

    r_df = pd.DataFrame(r_p_l, index=date_l)
    r_df.index = pd.to_datetime(r_df.index)
    # compute the std of the return for each month
    mean_df = r_df.resample('M').mean()
    std_df = r_df.resample('M').std() * np.sqrt(250 / horizon) * 100

    mean_rp = np.mean(r_p_l) * 250 / horizon * 100
    sigma_rp = np.std(r_p_l) * np.sqrt(250 / horizon) * 100

    turnover = np.mean(to_p_l[1:])
    dist_ture = np.mean(dist_l)
    return mean_rp, sigma_rp, turnover, dist_ture


def Result(ret_df, vech_df, version_name, model_name, universe, horizon, port=''):
    result_files = [i for i in files if '_pred.csv' in i and version_name in i and universe in i and f'F{horizon}' in i]

    result_files.sort()
    print(result_files)

    for (i, item) in enumerate(result_files):
        print(i, item)

    files_l = []

    if port == '1/N' or port == 'RC':
        result_files = result_files[:1]

    sum_df_l = []
    for filename in result_files:
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0)
        file_key_name = filename.split('-')[0].split('GHAR+iden')[1] + '-' + filename.split('-')[1].split('GHAR+iden')[1].split('_')[0]
        files_l.append(file_key_name.replace('+', ''))

        tmp_df_l = []
        for idx in range(horizon):
            metric_l = Loss(ret_df, vech_df, test_pred_df, port, idx, horizon)
            clms_l = ['mean', 'sigma', 'turnover', 'distance']

            sum_df = pd.DataFrame(metric_l[:4]).T
            sum_df.columns = clms_l
            tmp_df_l.append(sum_df)

        tmp_sum_df = pd.concat(tmp_df_l)
        tmp_sum_df.index = range(horizon)
        sum_df_l.append(tmp_sum_df.mean())

    all_sum_df = pd.concat(sum_df_l, axis=1).T
    all_sum_df.index = files_l
    print(all_sum_df)
    return all_sum_df


def rank_MCS(loss_df, pval_df, rank_files_l):
    loss_mean_df = loss_df.mean(0)
    rank_df = loss_mean_df.rank()
    pval_df = pd.DataFrame(pval_df, columns=['p-value'])
    pval_df['loss'] = loss_mean_df
    pval_df['rank'] = rank_df
    return pval_df.loc[rank_files_l, ['loss', 'rank', 'p-value']]


if __name__ == '__main__':
    data_name = 'DJIA'
    horizon = 1
    ret_df = load_ret(data_name, horizon)
    vech_df = load_data(data_name, horizon)
    files = os.listdir(sum_path)
    files.sort()

    ana_path = '/data01/Chao_Data/Forecast_Cov_20230708/Results_Analysis'

    rank_files_l = ['-', '-global', '-line', 'sector-', 'sector-global', 'sector-line',
                    'global-', 'global-global', 'global-line', 'glasso-', 'glasso-global', 'glasso-line']

    Result(ret_df, vech_df, 'Forecast_Cov', 'GHAR', universe, horizon, port='1/N')
    Result(ret_df, vech_df, 'Forecast_Cov', 'GHAR', universe, horizon, port='RC')

    all_sum_df, all_std_df = Result(ret_df, vech_df, 'Forecast_Cov', 'GHAR', data_name, horizon, port='')
    all_sum_df = all_sum_df.loc[rank_files_l]
    print(all_sum_df)
    all_sum_df.to_csv(join(path, 'Results_Analysis', f'{universe}_Cov_GMVPPlus_F{horizon}_Daily.csv'))