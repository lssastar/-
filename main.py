import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from abess import LinearRegression
import os
import argparse

def generate_simluated_data(n, p, Sigma = None, rho = None):
    # data generate
    if Sigma is None and rho is not None:
        Sigma = np.ones((p, p)) 
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = rho ** abs(i - j)

    if Sigma is None and rho is None:
        SystemError("Sigma and rho cannot be both None")
    
    np.random.seed(int(time.time()) % 5000)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, n)
    return X

### 部分后续用的函数
def generate_a_n(x, n):
    # 产生a_n
    a_n_out = x * n / (np.log(n)) **2
    return a_n_out

def aic(X, Y, beta):
    '''计算AIC
    '''
    n, p = X.shape
    loss = 2 * np.count_nonzero(beta) + n * np.log(np.linalg.norm(Y - np.dot(X, beta)) ** 2 / n)
    return loss

def bic(X, Y, beta):
    '''计算BIC
    '''
    n, p = X.shape
    loss = np.log(n) * np.count_nonzero(beta) + n * np.log(np.linalg.norm(Y - np.dot(X, beta)) ** 2 / n)
    return loss

def sic(X, Y, beta):
    '''计算SIC
    '''
    n, p = X.shape
    loss = np.log(np.log(n)) * np.count_nonzero(beta) * np.log(p) + n * np.log(np.linalg.norm(Y - np.dot(X, beta)) ** 2 / (2 * n))
    return loss


def Chen_Li(X, Y, beta_OLS, k_folds = 5, alpha_max = 2):
    '''aic计算Chen_Li'''
    n, p = X.shape
    diag = np.linalg.inv(np.dot(X.T, X))
    # generate k_fold splits of X and Y
    kf = KFold(n_splits = k_folds)
    kf.get_n_splits(X)

    alphas = np.linspace(0.0001, alpha_max, 100)

    ideal_alpha = 0
    sum_loss = np.inf
    for alpha in alphas:
        a_n = generate_a_n(alpha, n)
        loss = 0
        # 对每一fold 进行计算
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            J_n = np.where((beta_OLS ** 2)[:,0] >= a_n * np.diag(diag))[0]
            ## 取出J_n对应的X_train，Y_train
            X_train_J_n = X_train[:, J_n]
            beta_Chen_li = np.dot(np.linalg.inv(np.dot(X_train_J_n.T, X_train_J_n)), np.dot(X_train_J_n.T, Y_train))
            ## 构建新的beta,其中与J_n对于的值为beta_Chen_li，其余为0
            beta_Chen_li_new = np.zeros((p, 1))
            beta_Chen_li_new[J_n] = beta_Chen_li

            ## 计算beta_Chen_li_new对应的loss
            loss = loss + aic(X_test, Y_test, beta_Chen_li_new)
        if loss < sum_loss:
            sum_loss = loss
            ideal_alpha = alpha
    a_n = generate_a_n(ideal_alpha, n)
    J_n = np.where((beta_OLS ** 2)[:,0] >= a_n * np.diag(diag))[0]
    X_J_n = X[:, J_n]
    beta_Chen_li = np.dot(np.linalg.inv(np.dot(X_J_n.T, X_J_n)), np.dot(X_J_n.T, Y))
    beta_Chen_li_new = np.zeros((p, 1))
    beta_Chen_li_new[J_n] = beta_Chen_li
    return ideal_alpha, beta_Chen_li_new

def Chen_Li_bic(X, Y, beta_OLS, k_folds = 5, alpha_max = 2):
    '''bic计算Chen_Li'''
    n, p = X.shape
    diag = np.linalg.inv(np.dot(X.T, X))
    # generate k_fold splits of X and Y
    kf = KFold(n_splits = k_folds)
    kf.get_n_splits(X)

    alphas = np.linspace(0.0001, alpha_max, 100)

    ideal_alpha = 0
    sum_loss = np.inf
    for alpha in alphas:
        a_n = generate_a_n(alpha, n)
        loss = 0
        # 对每一fold 进行计算
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            J_n = np.where((beta_OLS ** 2)[:,0] >= a_n * np.diag(diag))[0]
            ## 取出J_n对应的X_train，Y_train
            X_train_J_n = X_train[:, J_n]
            beta_Chen_li = np.dot(np.linalg.inv(np.dot(X_train_J_n.T, X_train_J_n)), np.dot(X_train_J_n.T, Y_train))
            ## 构建新的beta,其中与J_n对于的值为beta_Chen_li，其余为0
            beta_Chen_li_new = np.zeros((p, 1))
            beta_Chen_li_new[J_n] = beta_Chen_li

            ## 计算beta_Chen_li_new对应的loss
            loss = loss + bic(X_test, Y_test, beta_Chen_li_new)
        if loss < sum_loss:
            sum_loss = loss
            ideal_alpha = alpha
    a_n = generate_a_n(ideal_alpha, n)
    J_n = np.where((beta_OLS ** 2)[:,0] >= a_n * np.diag(diag))[0]
    X_J_n = X[:, J_n]
    beta_Chen_li = np.dot(np.linalg.inv(np.dot(X_J_n.T, X_J_n)), np.dot(X_J_n.T, Y))
    beta_Chen_li_new = np.zeros((p, 1))
    beta_Chen_li_new[J_n] = beta_Chen_li
    return ideal_alpha, beta_Chen_li_new

def Chen_Li_sic(X, Y, beta_OLS, k_folds = 5, alpha_max = 2):
    '''sic计算Chen_Li'''
    n, p = X.shape
    diag = np.linalg.inv(np.dot(X.T, X))
    # generate k_fold splits of X and Y
    kf = KFold(n_splits = k_folds)
    kf.get_n_splits(X)

    alphas = np.linspace(0.0001, alpha_max, 100)

    ideal_alpha = 0
    sum_loss = np.inf
    for alpha in alphas:
        a_n = generate_a_n(alpha, n)
        loss = 0
        # 对每一fold 进行计算
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            J_n = np.where((beta_OLS ** 2)[:,0] >= a_n * np.diag(diag))[0]
            ## 取出J_n对应的X_train，Y_train
            X_train_J_n = X_train[:, J_n]
            beta_Chen_li = np.dot(np.linalg.inv(np.dot(X_train_J_n.T, X_train_J_n)), np.dot(X_train_J_n.T, Y_train))
            ## 构建新的beta,其中与J_n对于的值为beta_Chen_li，其余为0
            beta_Chen_li_new = np.zeros((p, 1))
            beta_Chen_li_new[J_n] = beta_Chen_li

            ## 计算beta_Chen_li_new对应的loss
            loss = loss + sic(X_test, Y_test, beta_Chen_li_new)
        if loss < sum_loss:
            sum_loss = loss
            ideal_alpha = alpha
    a_n = generate_a_n(ideal_alpha, n)
    J_n = np.where((beta_OLS ** 2)[:,0] >= a_n * np.diag(diag))[0]
    X_J_n = X[:, J_n]
    beta_Chen_li = np.dot(np.linalg.inv(np.dot(X_J_n.T, X_J_n)), np.dot(X_J_n.T, Y))
    beta_Chen_li_new = np.zeros((p, 1))
    beta_Chen_li_new[J_n] = beta_Chen_li
    return ideal_alpha, beta_Chen_li_new

parser = argparse.ArgumentParser()

parser.add_argument('--lown', type = int, default = 40, help = 'number of samples in low dimension')
parser.add_argument('--highn', type = int, default = 400, help = 'number of samples in high dimension')
parser.add_argument('--lowp', type = int, default = 8, help = 'number of features in low dimension')
parser.add_argument('--highp', type = int, default = 200, help = 'number of features in high dimension')
parser.add_argument('--lowrho', type = float, default = 0.8, help = 'correlation coefficient in low dimension')
parser.add_argument('--highrho', type = float, default = 0.8, help = 'correlation coefficient in high dimension')
parser.add_argument('--lowsigma', type = float, default = 1, help = 'standard deviation in low dimension')
parser.add_argument('--highsigma', type = float, default = 2, help = 'standard deviation in high dimension')
parser.add_argument('--lowdim', type = bool, default = True, help = 'whether to use low dimension data')
parser.add_argument('--highdim', type = bool, default = True, help = 'whether to use high dimension data')

args = parser.parse_args()

lown = args.lown
highn = args.highn
lowp = args.lowp
highp = args.highp
lowrho = args.lowrho
highrho = args.highrho
lowsigma = args.lowsigma
highsigma = args.highsigma
lowdim = args.lowdim
highdim = args.highdim


print(os.getcwd())
os.chdir("C:\\Users\\28676\\OneDrive\\文档\\vscode\\毕业论文项目\\my_codes")

#################################### low_dim
if lowdim:
    n = lown
    p = lowp
    myrho = lowrho
    mysigma = lowsigma
    beta = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
    beta = np.reshape(beta, (p, 1))

    active_set = np.array([1, 2, 5])
    inactive_set = np.array([3, 4, 6, 7, 8])

    X1 = generate_simluated_data(n, p, rho = myrho)
    # normalize data
    X1 = (X1 - np.mean(X1, axis = 0)) / np.std(X1, axis = 0)
    # generate Y
    Y1 = np.dot(X1, beta) + np.random.normal(0, scale = mysigma, size = (n, 1))

    beta_OLS = np.dot(np.linalg.inv(np.dot(X1.T, X1)), np.dot(X1.T, Y1))
    beta_OLS = np.reshape(beta_OLS, (p, 1))

    ### 下面是adapative lasso的代码
    X1_transformed = X1 * np.abs(beta_OLS.T)

    t_start = time.time()
    reg_aLASSO = LassoCV(cv = 5, random_state = 0).fit(X1_transformed, Y1)
    t_end = time.time()

    beta_aLASSO = reg_aLASSO.coef_ * np.abs(beta_OLS.T)
    beta_aLASSO = np.reshape(beta_aLASSO, (p, 1))

    active_set_LASSO = np.where(abs(beta_aLASSO) > 1e-5)[0] + 1
    inactive_set_LASSO = np.where(abs(beta_aLASSO) <= 1e-5)[0] + 1

    if len(active_set_LASSO) == 0:
        if len(active_set) == 0:
            TPR = 1
        else:
            TPR = 0
    else:
        TPR = len(np.intersect1d(active_set_LASSO, active_set)) / len(active_set_LASSO)
    
    if len(inactive_set_LASSO) == 0:
        if len(inactive_set) == 0:
            TNR = 1
        else:
            TNR = 0
    else:
        TNR = len(np.intersect1d(inactive_set_LASSO, inactive_set)) / len(inactive_set_LASSO)

    ReErr = np.linalg.norm(beta_aLASSO - beta) / np.linalg.norm(beta)

    Runtime = t_end - t_start

    out_dir1 = os.path.join(os.getcwd(), "output", "low_dim", f"aLASSO_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir1, "a") as f:
        f.write(f"{TPR},{TNR},{ReErr},{Runtime},{myrho},{mysigma},{n},aLASSO\n")

    ### 下面是abess的代码
    abess_model = LinearRegression()

    t_start = time.time()
    abess_model.fit(X1, Y1)
    t_end = time.time()

    beta_abess = abess_model.coef_
    beta_abess = np.reshape(beta_abess, (p, 1))

    active_set_abess = np.where(abs(beta_abess) > 1e-5)[0] + 1
    inactive_set_abess = np.where(abs(beta_abess) <= 1e-5)[0] + 1

    if len(active_set_abess) == 0:
        if len(active_set) == 0:
            TPR_abess = 1
        else:
            TPR_abess = 0
    else:
        TPR_abess = len(np.intersect1d(active_set_abess, active_set)) / len(active_set_abess)
    
    if len(inactive_set_abess) == 0:
        if len(inactive_set) == 0:
            TNR_abess = 1
        else:
            TNR_abess = 0
    else:
        TNR_abess = len(np.intersect1d(inactive_set_abess, inactive_set)) / len(inactive_set_abess)

    ReErr_abess = np.linalg.norm(beta_abess - beta) / np.linalg.norm(beta)

    Runtime_abess = t_end - t_start

    out_dir2 = os.path.join(os.getcwd(), "output", "low_dim", f"abess_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir2, "a") as f:
        f.write(f"{TPR_abess},{TNR_abess},{ReErr_abess},{Runtime_abess},{myrho},{mysigma},{n},ABESS\n")

    ### 下面是Chen-Li的代码
    t_start = time.time()
    ideal_alpha, beta_Chen_li = Chen_Li(X1, Y1, beta_OLS)
    t_end = time.time()

    beta_Chen_li = np.reshape(beta_Chen_li, (p, 1))

    active_set_Chen_li = np.where(abs(beta_Chen_li) > 1e-5)[0] + 1
    inactive_set_Chen_li = np.where(abs(beta_Chen_li) <= 1e-5)[0] + 1

    if len(active_set_Chen_li) == 0:
        if len(active_set) == 0:
            TPR_Chen_li = 1
        else:
            TPR_Chen_li = 0
    else:
        TPR_Chen_li = len(np.intersect1d(active_set_Chen_li, active_set)) / len(active_set_Chen_li)
    
    if len(inactive_set_Chen_li) == 0:
        if len(inactive_set) == 0:
            TNR_Chen_li = 1
        else:
            TNR_Chen_li = 0
    else:
        TNR_Chen_li = len(np.intersect1d(inactive_set_Chen_li, inactive_set)) / len(inactive_set_Chen_li)

    ReErr_Chen_li = np.linalg.norm(beta_Chen_li - beta) / np.linalg.norm(beta)

    Runtime_Chen_li = t_end - t_start
    
    out_dir3 = os.path.join(os.getcwd(), "output", "low_dim", f"Chen_Li_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir3, "a") as f:
        f.write(f"{TPR_Chen_li},{TNR_Chen_li},{ReErr_Chen_li},{Runtime_Chen_li},{myrho},{mysigma},{n},Chen_Li_aic,{ideal_alpha}\n")

    ### 下面是Chen-Li-bic的代码
    t_start = time.time()
    ideal_alpha_bic, beta_Chen_li_bic = Chen_Li_bic(X1, Y1, beta_OLS)
    t_end = time.time()

    beta_Chen_li_bic = np.reshape(beta_Chen_li_bic, (p, 1))

    active_set_Chen_li_bic = np.where(abs(beta_Chen_li_bic) > 1e-5)[0] + 1
    inactive_set_Chen_li_bic = np.where(abs(beta_Chen_li_bic) <= 1e-5)[0] + 1

    if len(active_set_Chen_li_bic) == 0:
        if len(active_set) == 0:
            TPR_Chen_li_bic = 1
        else:
            TPR_Chen_li_bic = 0
    else:
        TPR_Chen_li_bic = len(np.intersect1d(active_set_Chen_li_bic, active_set)) / len(active_set_Chen_li_bic)
    
    if len(inactive_set_Chen_li_bic) == 0:
        if len(inactive_set) == 0:
            TNR_Chen_li_bic = 1
        else:
            TNR_Chen_li_bic = 0
    else:
        TNR_Chen_li_bic = len(np.intersect1d(inactive_set_Chen_li_bic, inactive_set)) / len(inactive_set_Chen_li_bic)

    ReErr_Chen_li_bic = np.linalg.norm(beta_Chen_li_bic - beta) / np.linalg.norm(beta)

    Runtime_Chen_li_bic = t_end - t_start

    out_dir4 = os.path.join(os.getcwd(), "output", "low_dim", f"Chen_Li_bic_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir4, "a") as f:
        f.write(f"{TPR_Chen_li_bic},{TNR_Chen_li_bic},{ReErr_Chen_li_bic},{Runtime_Chen_li_bic},{myrho},{mysigma},{n},Chen_Li_bic,{ideal_alpha_bic}\n")

    ### 下面是Chen-Li-sic的代码
    t_start = time.time()
    ideal_alpha_sic, beta_Chen_li_sic = Chen_Li_sic(X1, Y1, beta_OLS)
    t_end = time.time()

    beta_Chen_li_sic = np.reshape(beta_Chen_li_sic, (p, 1))

    active_set_Chen_li_sic = np.where(abs(beta_Chen_li_sic) > 1e-5)[0] + 1
    inactive_set_Chen_li_sic = np.where(abs(beta_Chen_li_sic) <= 1e-5)[0] + 1

    if len(active_set_Chen_li_sic) == 0:
        if len(active_set) == 0:
            TPR_Chen_li_sic = 1
        else:
            TPR_Chen_li_sic = 0
    else:
        TPR_Chen_li_sic = len(np.intersect1d(active_set_Chen_li_sic, active_set)) / len(active_set_Chen_li_sic)
    
    if len(inactive_set_Chen_li_sic) == 0:
        if len(inactive_set) == 0:
            TNR_Chen_li_sic = 1
        else:
            TNR_Chen_li_sic = 0
    else:
        TNR_Chen_li_sic = len(np.intersect1d(inactive_set_Chen_li_sic, inactive_set)) / len(inactive_set_Chen_li_sic)

    ReErr_Chen_li_sic = np.linalg.norm(beta_Chen_li_sic - beta) / np.linalg.norm(beta)

    Runtime_Chen_li_sic = t_end - t_start

    out_dir5 = os.path.join(os.getcwd(), "output", "low_dim", f"Chen_Li_sic_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir5, "a") as f:
        f.write(f"{TPR_Chen_li_sic},{TNR_Chen_li_sic},{ReErr_Chen_li_sic},{Runtime_Chen_li_sic},{myrho},{mysigma},{n},Chen_Li_sic,{ideal_alpha_sic}\n")

#################################### high_dim
if highdim:
    p = highp
    n = highn
    myrho = highrho
    mysigma = highsigma
    beta = np.zeros(p)
    selected_set = np.random.choice(range(0, p), size = 12, replace = False)
    strong_set = selected_set[:4]
    middle_set = selected_set[4:8]
    weak_set = selected_set[8:]

    beta[strong_set] = np.random.uniform(low = 3, high = 4, size = 4)
    beta[middle_set] = np.random.uniform(low = 1, high = 2, size = 4)
    beta[weak_set] = np.random.uniform(low = 0, high = 0.5, size = 4)

    beta = np.reshape(beta, (p, 1))


    active_set = np.concatenate((strong_set, middle_set, weak_set)) + 1
    inactive_set = np.setdiff1d(range(0, p ), active_set) + 1

    X2 = generate_simluated_data(n, p, rho = myrho)
    # normalize X
    X2 = (X2 - np.mean(X2, axis = 0)) / np.std(X2, axis = 0)
    Y2 = np.dot(X2, beta) + np.random.normal(0, 1, size = (n, 1))

    beta_OLS = np.dot(np.linalg.inv(np.dot(X2.T, X2)), np.dot(X2.T, Y2))
    beta_OLS = np.reshape(beta_OLS, (p, 1))

    ### 下面是adapative lasso的代码
    X2_transformed = X2 * np.abs(beta_OLS.T)

    t_start = time.time()
    reg_aLASSO = LassoCV(cv = 5, random_state = 0).fit(X2_transformed, Y2)
    t_end = time.time()

    beta_aLASSO = reg_aLASSO.coef_ * np.abs(beta_OLS.T)
    beta_aLASSO = np.reshape(beta_aLASSO, (p, 1))

    active_set_LASSO = np.where(abs(beta_aLASSO) > 1e-5)[0] + 1
    inactive_set_LASSO = np.where(abs(beta_aLASSO) <= 1e-5)[0] + 1

    if len(active_set_LASSO) == 0:
        if len(active_set) == 0:
            TPR = 1
        else:
            TPR = 0
    else:
        TPR = len(np.intersect1d(active_set_LASSO, active_set)) / len(active_set_LASSO)
    
    if len(inactive_set_LASSO) == 0:
        if len(inactive_set) == 0:
            TNR = 1
        else:
            TNR = 0
    else:
        TNR = len(np.intersect1d(inactive_set_LASSO, inactive_set)) / len(inactive_set_LASSO)

    ReErr = np.linalg.norm(beta_aLASSO - beta) / np.linalg.norm(beta)

    Runtime = t_end - t_start

    out_dir1 = os.path.join(os.getcwd(), "output", "high_dim", f"aLASSO_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir1, "a") as f:
        f.write(f"{TPR},{TNR},{ReErr},{Runtime},{myrho},{mysigma},{n},aLASSO\n")

    ### 下面是abess的代码
    abess_model = LinearRegression()

    t_start = time.time()
    abess_model.fit(X2, Y2)
    t_end = time.time()

    beta_abess = abess_model.coef_
    beta_abess = np.reshape(beta_abess, (p, 1))

    active_set_abess = np.where(abs(beta_abess) > 1e-5)[0] + 1
    inactive_set_abess = np.where(abs(beta_abess) <= 1e-5)[0] + 1

    if len(active_set_abess) == 0:
        if len(active_set) == 0:
            TPR_abess = 1
        else:
            TPR_abess = 0
    else:
        TPR_abess = len(np.intersect1d(active_set_abess, active_set)) / len(active_set_abess)
    
    if len(inactive_set_abess) == 0:
        if len(inactive_set) == 0:
            TNR_abess = 1
        else:
            TNR_abess = 0
    else:
        TNR_abess = len(np.intersect1d(inactive_set_abess, inactive_set)) / len(inactive_set_abess)

    ReErr_abess = np.linalg.norm(beta_abess - beta) / np.linalg.norm(beta)

    Runtime_abess = t_end - t_start

    out_dir2 = os.path.join(os.getcwd(), "output", "high_dim", f"abess_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir2, "a") as f:
        f.write(f"{TPR_abess},{TNR_abess},{ReErr_abess},{Runtime_abess},{myrho},{mysigma},{n},ABESS\n")

    ### 下面是Chen-Li的代码
    t_start = time.time()
    ideal_alpha, beta_Chen_li = Chen_Li(X2, Y2, beta_OLS)
    t_end = time.time()

    beta_Chen_li = np.reshape(beta_Chen_li, (p, 1))

    active_set_Chen_li = np.where(abs(beta_Chen_li) > 1e-5)[0] + 1
    inactive_set_Chen_li = np.where(abs(beta_Chen_li) <= 1e-5)[0] + 1

    if len(active_set_Chen_li) == 0:
        if len(active_set) == 0:
            TPR_Chen_li = 1
        else:
            TPR_Chen_li = 0
    else:
        TPR_Chen_li = len(np.intersect1d(active_set_Chen_li, active_set)) / len(active_set_Chen_li)
    
    if len(inactive_set_Chen_li) == 0:
        if len(inactive_set) == 0:
            TNR_Chen_li = 1
        else:
            TNR_Chen_li = 0
    else:
        TNR_Chen_li = len(np.intersect1d(inactive_set_Chen_li, inactive_set)) / len(inactive_set_Chen_li)

    ReErr_Chen_li = np.linalg.norm(beta_Chen_li - beta) / np.linalg.norm(beta)

    Runtime_Chen_li = t_end - t_start

    out_dir3 = os.path.join(os.getcwd(), "output", "high_dim", f"Chen_Li_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir3, "a") as f:
        f.write(f"{TPR_Chen_li},{TNR_Chen_li},{ReErr_Chen_li},{Runtime_Chen_li},{myrho},{mysigma},{n},Chen_Li_aic,{ideal_alpha}\n")

    ### 下面是Chen-Li-bic的代码
    t_start = time.time()
    ideal_alpha, beta_Chen_li_bic = Chen_Li_bic(X2, Y2, beta_OLS)
    t_end = time.time()

    beta_Chen_li_bic = np.reshape(beta_Chen_li_bic, (p, 1))

    active_set_Chen_li_bic = np.where(abs(beta_Chen_li_bic) > 1e-5)[0] + 1

    if len(active_set_Chen_li_bic) == 0:
        if len(active_set) == 0:
            TPR_Chen_li_bic = 1
        else:
            TPR_Chen_li_bic = 0
    else:
        TPR_Chen_li_bic = len(np.intersect1d(active_set_Chen_li_bic, active_set)) / len(active_set_Chen_li_bic)
    
    if len(inactive_set_Chen_li_bic) == 0:
        if len(inactive_set) == 0:
            TNR_Chen_li_bic = 1
        else:
            TNR_Chen_li_bic = 0
    else:
        TNR_Chen_li_bic = len(np.intersect1d(inactive_set_Chen_li_bic, inactive_set)) / len(inactive_set_Chen_li_bic)

    ReErr_Chen_li_bic = np.linalg.norm(beta_Chen_li_bic - beta) / np.linalg.norm(beta)

    Runtime_Chen_li_bic = t_end - t_start

    out_dir4 = os.path.join(os.getcwd(), "output", "high_dim", f"Chen_Li_bic_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir4, "a") as f:
        f.write(f"{TPR_Chen_li_bic},{TNR_Chen_li_bic},{ReErr_Chen_li_bic},{Runtime_Chen_li_bic},{myrho},{mysigma},{n},Chen_Li_bic,{ideal_alpha}\n")

    ### 下面是Chen-Li-sic的代码
    t_start = time.time()
    ideal_alpha, beta_Chen_li_sic = Chen_Li_sic(X2, Y2, beta_OLS)
    t_end = time.time()

    beta_Chen_li_sic = np.reshape(beta_Chen_li_sic, (p, 1))

    active_set_Chen_li_sic = np.where(abs(beta_Chen_li_sic) > 1e-5)[0] + 1
    inactive_set_Chen_li_sic = np.where(abs(beta_Chen_li_sic) <= 1e-5)[0] + 1

    if len(active_set_Chen_li_sic) == 0:
        if len(active_set) == 0:
            TPR_Chen_li_sic = 1
        else:
            TPR_Chen_li_sic = 0
    else:
        TPR_Chen_li_sic = len(np.intersect1d(active_set_Chen_li_sic, active_set)) / len(active_set_Chen_li_sic)
    
    if len(inactive_set_Chen_li_sic) == 0:
        if len(inactive_set) == 0:
            TNR_Chen_li_sic = 1
        else:
            TNR_Chen_li_sic = 0
    else:
        TNR_Chen_li_sic = len(np.intersect1d(inactive_set_Chen_li_sic, inactive_set)) / len(inactive_set_Chen_li_sic)

    ReErr_Chen_li_sic = np.linalg.norm(beta_Chen_li_sic - beta) / np.linalg.norm(beta)

    Runtime_Chen_li_sic = t_end - t_start

    out_dir5 = os.path.join(os.getcwd(), "output", "high_dim", f"Chen_Li_sic_{myrho}_{mysigma}_{n}.csv")
    with open(out_dir5, "a") as f:
        f.write(f"{TPR_Chen_li_sic},{TNR_Chen_li_sic},{ReErr_Chen_li_sic},{Runtime_Chen_li_sic},{myrho},{mysigma},{n},Chen_Li_sic,{ideal_alpha}\n")