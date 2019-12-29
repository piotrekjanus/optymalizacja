from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
# Regularization for matrix inverting
MAT_REG = 1e-12

def update_save(history):
    #TODO
    pass


def lars(x, y):
    assert x is not None, "x should not be empty"
    assert y is not None, "y should not be empty"
    assert y.ndim == 2, "y is expected to be 2D array"

    # reg_factor = []
    # stop_condition = []

    # xtx =  np.dot(x.T, x) 
    
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
    mu_a = np.zeros((n,1))
    # Then the next step of the LARS algorithm updates µ
    # mu_a_plus = 0
    # mu_a_OLS = 0 
       
    # to liczymy
    beta = np.zeros((x.shape[1], 1))
    # beta_new = beta
    # beta_OLS = beta
    
    params = ["active_set", "add", "drop", "beta_norm", "beta", "b", "mu", "beta_ols_norm", "beta_ols",
              "b_ols", "mu_ols", "mse", "r_square"]
    
    history = dict()
    for key in params:
        history[key] = []
    
    history["beta"] = [np.zeros((m, 1))]
    history["b"] = my
    history["mu"] = my
    history["r_square"] = my
    history["b_ols"] = my
    history["mu_ols"] = my * np.ones(y.shape)
    history["mse"] = [np.power(res, 2).mean()]
    
    c = 0
    
    # max(abs(c))
    c_max = 0

    i = 0
    while i<100:
#        check_criterions()
        
        # eq. 2.8
        # vector of current correlations
        c = x.T.dot(res - mu_a)

        # eq. 2.9
        c_max_temp = np.max(np.abs(c[inactive_set]))
        c_max, c_max_ind = c_max_temp, np.where(np.abs(c) == c_max_temp)[0]

        # print(c_max_ind + 1)
        # eq. 2.9 
        # active set A is the set of indices
        # corresponding to covariates with the greatest absolute current correlations
        active_set = np.append(active_set, c_max_ind).astype(dtype = np.int16)
        inactive_set = np.setdiff1d(possible_var, active_set)
        
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
            i = 100
            
        d = np.zeros((m, 1))
        d[active_set, :] = s * wa

        # eq 2.12 2.19 2.21
        mu_a = mu_a + gamma * ua
        beta = beta + gamma * d 
        
        # mu_a = mu_a + c_max/Aa * ua 
        # beta = beta + c_max/Aa * d 
        MSE = np.sum((res - mu_a)**2)/len(res)
        # MSE = np.sum((res - x.dot(beta_new.T))**2)/len(res)
        # mu_a = mu_a_plus
        # beta = beta_new
        history["mse"] = history["mse"] + [MSE]
        history["beta"] = history["beta"] + [beta] 
        i += 1
    print(history["mse"])
    d = pd.DataFrame(np.round(history["beta"], 2).reshape((11, 10)))
    print(d)

    return history["beta"], x, y
        
if __name__ == "__main__":
    x, y = load_diabetes(return_X_y = True)
    # scaler = StandardScaler().fit(x)
    # x = scaler.transform(x)
    y = y.reshape((-1,1))
    # y = StandardScaler().fit_transform(y)
    print("Our results")
    lars(x, y) 

    from sklearn import linear_model
    lar = linear_model.Lars()
    model = lar.fit(x,y)
    print("Final coeff using Python implementation")
    print(model.coef_)
    print("(same as using R)")

# R output from lars(x, y)
# can be compared with our results

#         age        sex       bmi       map        tc       ldl       hdl      tch      ltg      glu
# 0    0.0000    0.00000   0.00000   0.00000    0.0000   0.00000    0.0000   0.0000   0.0000  0.00000
# 1    0.0000    0.00000  60.11927   0.00000    0.0000   0.00000    0.0000   0.0000   0.0000  0.00000
# 2    0.0000    0.00000 361.89461   0.00000    0.0000   0.00000    0.0000   0.0000 301.7753  0.00000
# 3    0.0000    0.00000 434.75796  79.23645    0.0000   0.00000    0.0000   0.0000 374.9158  0.00000
# 4    0.0000    0.00000 505.65956 191.26988    0.0000   0.00000 -114.1010   0.0000 439.6649  0.00000
# 5    0.0000  -74.91651 511.34807 234.15462    0.0000   0.00000 -169.7114   0.0000 450.6674  0.00000
# 6    0.0000 -111.97855 512.04409 252.52702    0.0000   0.00000 -196.0454   0.0000 452.3927 12.07815
# 7    0.0000 -197.75650 522.26485 297.15974 -103.9462   0.00000 -223.9260   0.0000 514.7495 54.76768
# 8    0.0000 -226.13366 526.88547 314.38927 -195.1058   0.00000 -152.4773 106.3428 529.9160 64.48742
# 9    0.0000 -227.17580 526.39059 314.95047 -237.3410  33.62827 -134.5994 111.3841 545.4826 64.60667
# 10 -10.0122 -239.81909 519.83979 324.39043 -792.1842 476.74584  101.0446 177.0642 751.2793 67.62539