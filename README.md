# PostCE
> Lu, Z., Geng, Z., Li, W., Zhu, S., & Jia, J. (2022). Evaluating Causes of Effects by Posterior Effects of Causes. Biometrika. https://doi.org/10.1093/biomet/asac038

Simple python code to calculate Posterior Total Causal Effect (PostTCE) and Posterior Direct Causal Effect (PostDCE).

## Description

Variables: cause variables $X=(X_{0},...,X_{p-1})$ and effect variable $Y$

`k`: int, $\{0,1,...,p-1\}$

`Obs`: np.array, observed evidence $[x_0,...,x_{p-1},y]$ where y=1 and x_{i} in {0,1,np.nan}

`Pr_joint`: numpy.ndarray, joint probability, `Pr_joint[x[0],...,x[p-1],y]` = $\Pr(X_{0}=x_0,...,X_{p-1}=x_{p-1},Y=y)$
