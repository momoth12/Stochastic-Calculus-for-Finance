import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
N=1000
sigma_0=0.5
lmda=3
gamma=0.5
T=1
c=0.5

M=1
def simulate_Ornstein_Uhlenbeck(sigma_0,lmda,gamma,T,c,n,M):
    dT=T/n
    dWt= np.zeros((n,M))
    dWt[1:] = np.random.randn(n-1, M)*np.sqrt(dT)
    interval=np.linspace(0,T,n)
    integral= np.exp(-lmda*interval)*np.cumsum(np.exp(lmda*interval)*dWt, axis=0)

    sigma_t=np.transpose(c + (sigma_0 - c)*np.exp(-lmda*interval) + gamma*integral)
    return sigma_t
n = 1000
sigma_0 = 1.0
def risky_asset_price(S0, sigma_0, lmda, c, gamma, T, n, N, sigma_traj=None):

    if sigma_traj is None:
        sigma_traj = simulate_Ornstein_Uhlenbeck(sigma_0, c, lmda, gamma, T, n-1, M=1000).transpose()
    else:
        N = sigma_traj.shape[0]
        sigma_traj = sigma_traj[:,:-1].transpose()
    dt = T/n
    times = np.linspace(0, T, n, endpoint=True)
    volatility_integral = np.zeros((n, N))
    volatility_integral[1:] = np.cumsum(np.power(sigma_traj, 2)*dt, axis=0)
    dwt = np.sqrt(dt)*np.random.randn(n-1, N)
    stoch_integral = np.zeros((n, N))
    stoch_integral[1:] = np.cumsum(sigma_traj*dwt, axis=0)
    result= S0*np.exp(-0.5*volatility_integral + stoch_integral).transpose()
    return result
n = 1000
So=100
time = np.linspace(0, T, n, endpoint=True)
def d_plus(s,k,v):
    return np.log(s/k)/np.sqrt(v) + np.sqrt(v)/2

def compute_greek_gamma(t, S0, sigm, K, r, tmax):

    dp = d_plus(S0, K*np.exp(-r*tmax) , sigm**2*tmax)
    return stats.norm.pdf(dp)/(S0*sigm*np.sqrt(tmax - t))
def prof_and_loss(sigm, r, K, S0, sig0, c, lmda, gamma, tmax, n, N):

    dt = tmax/n
    
    sigma_traj = simulate_Ornstein_Uhlenbeck(sig0,lmda, gamma, tmax,c, n, M=1000)
    
    
    asset_traj = risky_asset_price(S0, sig0,lmda,c, gamma, tmax, n,N=1000, sigma_traj=sigma_traj)
    asset_traj = asset_traj.T[:-1]
    sigma_traj = sigma_traj.T[:-1]
    asset_traj = asset_traj[:,:,None]
    sigma_traj = sigma_traj[:,:,None]
    times = np.linspace(0, tmax, n)[:-1]
    greek_gamma = compute_greek_gamma(times[:,None,None], asset_traj, sigm, K[None,None,:], r, tmax)
    function_ = np.exp(r*(tmax - times[:,None,None]))*(sigm**2 - sigma_traj**2)*(asset_traj**2)*greek_gamma
    integral_ =  np.sum(function_*dt, axis=0)
    return sigma_traj, 0.5*integral_.T
S0 = 100
tmax = 1
sig0 = 0.4
lmda = 2.0
c = 0.4
gamma = 0.3
sigm = sig0
r = 0.02
K_range = np.array([90, 95, 100, 105, 110])  # Add the strike prices you want to test

n_range = range(10,100,10)
n = 1001
N = 1000
samples_matrix = np.empty((6, len(K_range), N))
vol, pnl_sample = prof_and_loss(sigm, r, K_range, S0, sig0, c, lmda, gamma, tmax, n, N)

print(n)