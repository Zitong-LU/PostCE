
import numpy as np
import itertools as itt

'''
Variables: X=(X_{0},...,X_{p-1}) and Y

`k`: int, {0,1,...,p-1}

`Obs`: np.array, observed evidence [x_0,...,x_{p-1},y] where y=1 and x_{i} in {0,1,np.nan}

`Pr_joint`: numpy.ndarray, joint probability, Pr_joint[x[0],...,x[p-1],y] = Pr(X_{0}=x_0,...,X_{p-1}=x_{p-1},Y=y)
'''
# PostTCE(X_k \Rightarray Y \mid Obs)
def PostTCE(k,Obs,Pr_joint):
    def PostTCE_obs(k,Obs,Pr_joint):
        if Obs[-1]==0:
            return 0
        if Obs[k]==0:
            return 0
        else: 
            p = len(Obs) - 1
            dk = Obs[k+1:p]
            S = 0 
            if k==(p-1):
                c = Obs.copy()
                c[k] = 0 
                c[p] = 1
                S = (np.sum(Pr_joint[tuple(c)])/np.sum(Pr_joint[tuple(c[0:p])]))
            else: 
                for dk_ in itt.product([0,1],repeat=p-k-1):
                    if np.sum(np.array(dk_)>dk)>0:
                        continue
                    else: 
                        S1 = np.ones(p)
                        c = Obs.copy()
                        c[k] = 0 
                        c[k+1:p] = dk_
                        c[p] = 1
                        for i in range(k+1,p):
                            if Obs[i]==0:
                                S1[i] = 1
                            else:
                                idx1 = np.copy(c[0:i+1])
                                idx1[-1] = 1
                                idx2 = np.copy(Obs[0:i+1])
                                S1[i] = 1-c[i] + np.power(-1,1-c[i]) * (np.sum(Pr_joint[tuple(idx1)])/np.sum(Pr_joint[tuple(idx1[0:i])])) / (np.sum(Pr_joint[tuple(idx2)])/np.sum(Pr_joint[tuple(idx2[0:i])]))
                        S = S+ np.prod(S1) * (np.sum(Pr_joint[tuple(c)])/np.sum(Pr_joint[tuple(c[0:p])]))
            return (1-S / (np.sum(Pr_joint[tuple(Obs)])/np.sum(Pr_joint[tuple(Obs[0:p])])) )
    
    idx = np.isnan(Obs)
    l = np.sum(idx) 
    if l==0:
        return PostTCE_obs(k,Obs,Pr_joint)
    else: 
        S = 0
        S0 = 0
        for Obs_c in itt.product([0,1],repeat=l):
            Obs_full = Obs.copy()
            Obs_full[idx] = np.copy(Obs_c)
            Obs_full = np.array(Obs_full,dtype=int)
            S = S+ PostTCE_obs(k,Obs_full,Pr_joint) * Pr_joint[tuple(Obs_full)]
            S0 = S0+ Pr_joint[tuple(Obs_full)]
        return (S/S0)



'''
Variables: X=(X_{0},...,X_{p-1}) and Y

`k`: int, {0,1,...,p-1}

`dk_`: np.array, length=p-k-1

`Obs`: np.array, observed evidence [x_0,...,x_{p-1},y] where y=1 and x_{i} in {0,1,np.nan}

`Pr_joint`: numpy.ndarray, joint probability, Pr_joint[x[0],...,x[p-1],y] = Pr(X_{0}=x_0,...,X_{p-1}=x_{p-1},Y=y)
'''
# PostTCE(X_k \Rightarray Y_{d_k^*} \mid Obs)
def PostDCE(k,dk_,Obs,Pr_joint):
    def PostDCE_obs(k,dk_,Obs,Pr_joint):
        if Obs[-1]==0:
            return 0
        if Obs[k]==0:
            return 0
        else: 
            p = len(Obs)-1
            if k==p-1: 
                idx1 = np.copy(Obs)
                idx0 = np.copy(Obs)
                idx1[k] = 1
                idx0[k] = 0
                S = np.sum(Pr_joint[tuple(idx1)])/np.sum(Pr_joint[tuple(idx1[0:p])]) - np.sum(Pr_joint[tuple(idx0)])/np.sum(Pr_joint[tuple(idx0[0:p])])
                S = S / (np.sum(Pr_joint[tuple(Obs)])/np.sum(Pr_joint[tuple(Obs[0:p])]))
                return S
            else: 
                idx1 = np.copy(Obs)
                idx1[k+1:p] = dk_
                idx0 = np.copy(idx1)
                idx1[k] = 1
                idx0[k] = 0
                S = np.sum(Pr_joint[tuple(idx1)])/np.sum(Pr_joint[tuple(idx1[0:p])]) - np.sum(Pr_joint[tuple(idx0)])/np.sum(Pr_joint[tuple(idx0[0:p])])
                S = S / (np.sum(Pr_joint[tuple(Obs)])/np.sum(Pr_joint[tuple(Obs[0:p])]))
                return S
    
    idx = np.isnan(Obs)
    l = np.sum(idx) 
    if l==0:
        return PostDCE_obs(k,dk_,Obs,Pr_joint)
    else: 
        S = 0
        S0 = 0
        for Obs_c in itt.product([0,1],repeat=l):
            Obs_full = Obs.copy()
            Obs_full[idx] = np.copy(Obs_c)
            Obs_full = np.array(Obs_full,dtype=int)
            S = S+ PostDCE_obs(k,dk_,Obs_full,Pr_joint) * Pr_joint[tuple(Obs_full)]
            S0 = S0+ Pr_joint[tuple(Obs_full)]
        return (S/S0)
