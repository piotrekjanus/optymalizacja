from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
# Regularization for matrix inverting
MAT_REG = 1e-12
# Resolution for Lars
LARS_RES = 1e-12

def plot_path(beta_path):
    sum_abs_coeff = np.sum(np.abs(beta_path), 1)
    plt.plot(sum_abs_coeff, beta_path)
    plt.show()

def update_save(history):
    #TODO
    pass

def check_stop_criterions(params):
    force_stop = False
    # if np.abs(params["c_max"][-1] / params["Aa"][-1] - np.min([params["gamma"][-1], params["gamma_tilde"][-1]])) < LARS_RES:
    #     print("Exact matching")
    #     force_stop = True
    if params["mse"][-1] > params["mse"][-2]:
        print("MSE increased")
        force_stop = True
    if len(params["inactive_set"][-1]) == 0: 
        force_stop = True
    return force_stop


def lars(x, y, alg_type = "lars"):

    assert x is not None, "x should not be empty"
    assert y is not None, "y should not be empty"
    assert y.ndim == 2, "y is expected to be 2D array"
    assert alg_type.lower() in ["lasso", "lars"], 'only ["lars", "lar"] types are available, please select right type'
    # reg_factor = []
    # stop_condition = []

    my = np.mean(y)

    # Calculate residuals
    res = y - my
    
    n = x.shape[0]
    m = x.shape[1]    

    var_col = np.var(x, axis = 0)
    
    # set of indexes
    active_set = []
    
    # set of indexes
    possible_var = np.where(var_col > 0)
    inactive_set = np.where(var_col > 0)

    # Suppose that µA is the current LARS estimate 
    mu_a = np.zeros((n, 1))
       
    beta = np.zeros((x.shape[1], 1))
    
    params = ["active_set", "c_max", "Aa", "gamma", "gamma_tilde"]
    
    history = dict()
    for key in params:
        history[key] = []
    
    history["beta"] = [np.zeros((m, 1))]
    history["mse"] = [np.power(res, 2).mean()]
    history["inactive_set"] = [inactive_set]
    
    c = 0
    
    # max(abs(c))
    c_max = 0

    # used for lasso version od LARS
    drop = []  

    i = 0
    while i<12:
        
        # eq. 2.8
        # vector of current correlations
        c = x.T.dot(res - mu_a)

        # eq. 2.9
        c_max_temp = np.max(np.abs(c[inactive_set]))
        c_max, c_max_ind = c_max_temp, np.where(np.abs(c) == c_max_temp)[0]

        # eq. 2.9 
        # active set A is the set of indices
        # corresponding to covariates with the greatest absolute current correlations
        active_set = np.append(active_set, c_max_ind).astype(dtype = np.int16)
        inactive_set = np.setdiff1d(possible_var, active_set)

        # LASSO
        if alg_type == 'lasso':
            if len(drop) > 0 and len(np.where(drop == c_max_ind)) == 0:
                del active_set[active_set == c_max_ind]

            if len(drop) > 0:
                active_set = np.setdiff1d(active_set, drop)
                inactive_set = np.unique(np.append(inactive_set, drop))

        # eq. 2.10
        s = np.sign(c[active_set])

        # eq. 2.4 
        xa = x[:, active_set] 
        xa *= s.reshape(-1)
        
        # eq. 2.5
        ga = xa.T.dot(xa)

        ga  = ga + np.eye(len(ga)) * MAT_REG
        # Alternatively  inv_ga = np.linalg.pinv(ga)  can be applied
        inv_ga, _, _, _ = np.linalg.lstsq(ga, np.eye(ga.shape[0]), rcond=None)
       
        ones_vec = np.ones((len(active_set), 1))
        # 1'*Ga*1
        # eq. 2.5
        # Alternatively Aa = np.sqrt(1/np.sum(np.sum(inv_ga)))
        Aa = np.sqrt(1/ones_vec.T.dot(inv_ga).dot(ones_vec))

        # eq 2.6
        # Alternatively wa = Aa * np.sum(inv_ga, axis=1)
        wa = Aa * inv_ga.dot(ones_vec)
        # ua: equiangular vector is the unit vector making equal angles, less than 90◦
        ua = xa.dot(wa)

        # check conditions that ua is well defined 
        assert np.all(xa.T.dot(ua) > (Aa*np.ones((1, n)) - xa.T.dot(ua) * 0.01)), "Not working for iteration {}".format(i)
        assert np.all(xa.T.dot(ua) < (Aa*np.ones((1, n)) + xa.T.dot(ua) * 0.01)), "Not working for iteration {}".format(i)
        assert np.linalg.norm(ua, 2) >  0.999, "Not working for iteration {}".format(i)
        assert np.linalg.norm(ua, 2) <  1.001, "Not working for iteration {}".format(i)

        # eq. 2.11
        a = x.T.dot(ua)

        # eq 2.13
        gamma_1 = (c_max - c[inactive_set])/(Aa - a[inactive_set])
        gamma_2 = (c_max + c[inactive_set])/(Aa + a[inactive_set])
        
        gamma_ = np.append(gamma_1, gamma_2)

        if len(gamma_[gamma_ > 0]) > 0:
            gamma = np.min(gamma_[gamma_ > 0])
        else:
            gamma = c_max/Aa

        
        d = np.zeros((m, 1))
        d[active_set, :] = s * wa

        tmp = np.zeros((m, 1))
        # eq 3.4
        tmp[active_set, :] = -1 * beta[active_set] / d[active_set, :] 
        tmp2 = tmp[tmp > 0]
        
        drop = []
        # eq 3.5
        gamma_tilde = np.inf
        if len(tmp2) > 0 and gamma >= np.min(tmp2):
            # eq 3.5
            gamma_tilde = np.min(tmp2)
            # eq 3.6 
            drop, _ = np.where(tmp == gamma_tilde)
            
        if alg_type == 'lars':
            mu_a_ = mu_a + gamma * ua
            beta_ = beta + gamma * d
            drop = []

        if alg_type == 'lasso':
            mu_a_ = mu_a + np.min([gamma, gamma_tilde]) * ua
            beta_ = beta + np.min([gamma, gamma_tilde]) * d
            active_set = np.setdiff1d(active_set, drop)
            inactive_set = np.setdiff1d(possible_var, active_set)
            beta_[drop] = 0

        # eq 2.12 2.19 2.21
        mu_a = mu_a_
        beta = beta_ 
        
        # mu_a = mu_a + c_max/Aa * ua 
        # beta = beta + c_max/Aa * d 
        MSE = np.sum((res - mu_a)**2)/len(res)
        # MSE = np.sum((res - x.dot(beta_new.T))**2)/len(res)
        # mu_a = mu_a_plus
        # beta = beta_new
        history["mse"] = history["mse"] + [MSE]
        history["beta"] = history["beta"] + [beta] 
        history["gamma"] = history["gamma"] + [gamma]
        history["gamma_tilde"] = history["gamma_tilde"] + [gamma]
        history["Aa"] = history["Aa"] + [Aa]
        history["c_max"] = history["c_max"] + [c_max]
        history["inactive_set"] = [inactive_set]

        i += 1
        if check_stop_criterions(history):
            break

    print(history["mse"])
    d = pd.DataFrame(np.round(history["beta"], 2).reshape((i+1, 10)))
    print(d)

    return np.squeeze(history["beta"])
        
if __name__ == "__main__":
    x, y = load_diabetes(return_X_y = True)
    # scaler = StandardScaler().fit(x)
    # x = scaler.transform(x)
    y = y.reshape((-1,1))
    # y = StandardScaler().fit_transform(y)
    # beta_path = lars(x, y, 'lars') 
    # plot_path(beta_path)
    beta_path = lars(x, y, 'lasso')
    plot_path(beta_path)