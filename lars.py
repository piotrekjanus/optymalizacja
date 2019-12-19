from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_diabetes

def update_save(history):
    #TODO
    pass


def lars(x = None, y = None):
    reg_factor = []
    stop_condition = []
    
    x, y = load_diabetes(return_X_y = True)

    y = y.reshape((-1,1))
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    y = StandardScaler().fit_transform(y)
    # xtx =  np.dot(x.T, x) 
    
    my = np.mean(y)
    res = y - my
    
    n = x.shape[0]
    m = x.shape[1]
    
    var_col = np.var(x, axis = 0)
    
    # set of indexes
    active_set = []
    
    # set of indexes
    possible_var = np.where(var_col > 0)
    inactive_set = np.where(var_col > 0)
    #Suppose that µA is the current LARS estimate and that
    mu_a        = np.zeros((n,1))
    # Then the next step of the LARS algorithm updates µ
    mu_a_plus   = 0
    
    mu_a_OLS    = 0 
       
    # to liczymy
    beta        = np.zeros((1, x.shape[1]))
    beta_new    = beta
    beta_OLS    = beta
    
    params = ["active_set", "add", "drop", "beta_norm", "beta", "b", "mu", "beta_ols_norm", "beta_ols",
              "b_ols", "mu_ols", "mse", "r_square"]
    
    history = dict()
    for key in params:
        history[key] = []
        
    history["b"] = my
    history["mu"] = my
    history["r_square"] = my
    history["b_ols"] = my
    history["mu_ols"] = my * np.ones(y.shape)
    history["mse"] = [np.power(res, 2).mean()]
    
    c = 0
    
    # max(abs(c))
    c_max = 0
    
    C_max_ind       = []
    C_max_ind_pl    = []
    drop            = []
    i = 0
    while i<100:
#        check_criterions()
        
        c = x.T.dot(res - mu_a)
        c_max_temp = np.max(c[inactive_set])
        c_max, c_max_ind = c_max_temp, np.where(c == c_max_temp)[0]
        
        active_set = np.sort(np.append(active_set, c_max_ind)).astype(dtype = np.int16) 
        inactive_set = np.setdiff1d(possible_var, active_set)
        
        s = np.sign(c[active_set])
        
        xa = x[:, active_set] * np.repeat(s.T, n).reshape((n, len(active_set)))
        
        ga = xa.T.dot(xa)
        ga  = ga+np.eye(len(ga))*1e-11
        
        #inv_ga = np.linalg.pinv(ga)
        inv_ga,_,_,_ = np.linalg.lstsq(ga, np.eye(ga.shape[0]))
        # 1'*Ga*1
        
        
        # Aa = np.sqrt(1/np.ones(len(c)).dot(inv_ga).dot(np.ones(len(c)).T))
        Aa= np.sqrt(1/np.sum(np.sum(inv_ga)))
        
        wa = Aa * np.sum(inv_ga, axis=1)
        ua = xa.dot(wa).reshape((-1,1))
        
        # xa.T.dot(ua) == Aa*np.ones((1, n))
        # np.linalg.norm(ua, 2)
        
        a = x.T.dot(ua)
        
        gamma_1 = (c_max - c[inactive_set])/(Aa - a[inactive_set])
        gamma_2 = (c_max + c[inactive_set])/(Aa + a[inactive_set])
        
        gamma_ = np.append(gamma_1, gamma_2)
        if len(gamma_[gamma_ > 0]) > 0:
            gamma = np.min(gamma_[gamma_ > 0])
        else:
            gamma = c_max/Aa
            i = 100
            
        ## lepiej to sprawdzić
        d = np.zeros((1,m))
        d[:, active_set] = (s.T*wa)
        mu_a_plus = mu_a + gamma*ua
        beta_new = beta + gamma*d 
        # drop = []
        
        mu_a_OLS    = mu_a + c_max/Aa*ua #          % eq 2.19, 2.21
        beta_OLS    = beta + c_max/Aa*d #           % eq 2.19, 2.21
        # MSE         = np.sum((res - mu_a_OLS)**2)/len(res)
        MSE = np.sum((res - x.dot(beta_new.T))**2)/len(res)
        mu_a = mu_a_plus
        beta = beta_new
        history["mse"] = history["mse"] + [MSE]
        history["beta"] = history["beta"] + [beta] 
        i += 1
        print(i)
    print(history["mse"])
    # print(history["beta"])

    return history["beta"], x, y
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bety, x, y = lars() 
    for beta in bety:
        
        plt.plot(y)
        plt.plot(x.dot(beta.T))
        plt.show()
        input()


