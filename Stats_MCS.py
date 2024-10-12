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


def lower2vech(A):
    # stack the lower triangle part of a systematic matrix
    L = np.tril(A)
    n = A.shape[0]
    vech = []
    for i in range(n):
        vech.extend(L[i, :i+1])

    return np.array(vech)


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


def Loss(vech_df, test_pred_df):
    date_l = list(test_pred_df.index)
    test_vech_df = vech_df.loc[date_l]
    df_l = []

    for i, date in enumerate(date_l):
        if i % 250 == 0:
            print(date)
        pred_vech = test_pred_df.loc[date]
        test_vech = test_vech_df.loc[date]

        H_t = square_cholesky(pred_vech)

        if is_pos_def(H_t):
            pass
        else:
            pre_date = date_l[date_l.index(date) - 1]
            pre_vech = test_vech_df.loc[pre_date]
            H_t = square_cholesky(pre_vech)

        Sigma_t = square_cholesky(test_vech)

        # Sigma_t is the real covariance matrix; H_t is its prediction
        diff_sigma = Sigma_t - H_t

        diff_vech = lower2vech(diff_sigma)
        e = np.dot(diff_vech, diff_vech.T)

        f = np.linalg.norm(diff_sigma)
        # f = np.sqrt(np.trace(np.dot(diff_sigma, diff_sigma)))

        q = np.log(np.linalg.det(H_t)) + np.trace(np.dot(np.linalg.inv(H_t), Sigma_t))

        df_l.append([np.sqrt(e), f, q])

    df = pd.DataFrame(np.array(df_l), index=date_l, columns=['L_E', 'L_F', 'L_Q'])

    return df


def Result(vech_df, version_name, data_name, horizon):
    result_files = [i for i in files if '_pred.csv' in i and version_name in i and universe in i and f'F{horizon}' in i]
    result_files.sort()
    print(result_files)

    E_df_l = []
    F_df_l = []
    Q_df_l = []
    files_l = []

    for filename in result_files:
        print(filename)
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0)

        df = Loss(vech_df, test_pred_df)

        E_df_l.append(df['L_E'])
        F_df_l.append(df['L_F'])
        Q_df_l.append(df['L_Q'])

        try:
            file_key_name = filename.split('-')[0].split('GHAR+iden')[1] + '-' + filename.split('-')[1].split('GHAR+iden')[1].split('_')[0]
        except:
            file_key_name = filename.split('-')[0].split('GHAR+iden')[1] + '-' + filename.split('-')[1].split('GHAR')[1].split('_')[0]
        print(file_key_name)
        # file_key_name = filename.split('-')[0].split('+')[1] + '-' + filename.split('-')[1].split('+')[1].split('_')[0]
        files_l.append(file_key_name.replace('+', ''))

    E_df = pd.concat(E_df_l, axis=1)
    E_df.columns = files_l
    F_df = pd.concat(F_df_l, axis=1)
    F_df.columns = files_l
    Q_df = pd.concat(Q_df_l, axis=1)
    Q_df.columns = files_l
    return E_df, F_df, Q_df


def rank_MCS(loss_df, pval_df, rank_files_l):
    loss_mean_df = loss_df.mean(0)
    rank_df = loss_mean_df.rank()
    pval_df = pd.DataFrame(pval_df, columns=['p-value'])
    pval_df['loss'] = loss_mean_df
    pval_df['ratio'] = loss_mean_df / loss_mean_df.loc['-']
    pval_df['rank'] = rank_df
    return pval_df.loc[rank_files_l, ['loss', 'ratio', 'rank', 'p-value']]


if __name__ == '__main__':
    data_name = 'DJIA'
    horizon = 1
    vech_df = load_data(data_name, horizon)
    files = os.listdir(sum_path)
    files.sort()

    rank_files_l = ['-', '-global', '-line', 'sector-', 'sector-global', 'sector-line',
                    'global-', 'global-global', 'global-line', 'glasso-', 'glasso-global', 'glasso-line']

    E_df, F_df, Q_df = Result(vech_df, 'Forecast_Cov', data_name, horizon)
    print(E_df.mean(0))
    print(F_df.mean(0))
    print(Q_df.mean(0))

    BL = 2
    mcs_E = ModelConfidenceSet(E_df, 0.05, 10000, BL, 'SQ').run()
    sum_E = rank_MCS(E_df, mcs_E.pvalues, rank_files_l)