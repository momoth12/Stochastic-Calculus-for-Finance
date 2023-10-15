M=1000
n=1000
T=2
deltaT=T/n
samples_I=[]
samples_J=[]
samples_K=[]
for i in range(M):
    W=np.cumsum(np.sqrt(deltaT)*np.random.randn(n+1))
    deltaW=np.diff(W)
    a=0.5*(W[-1]**2)-np.sum(W[:-1]*deltaW)
    b=0.5*(W[-1]**2)-np.sum(W[1:]*deltaW)
    c=0.5*(W[-1]**2)-0.5*np.sum((W[1:]+W[:-1])*deltaW)
    samples_I.append(a)
    samples_J.append(b)
    samples_K.append(c)
    
