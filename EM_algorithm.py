import numpy as np
import random
from scipy.stats import multivariate_normal as mvn
def EM_gmm(Xdata,prob_z,mu,sigma):
    n,l=Xdata.shape
    k=len(prob_z)
    lihood_old=0
    
    for i in range(100):
        lihood_new=0
        w=np.zeros((k,n))
        for j in range(k):
            for i in range(n):
                w[j,i]=prob_z[j] * mvn(mu[j], sigma[j]).pdf(Xdata[i])
                #print(w)
                #print(w)
        w/=w.sum(0)
        
        
        mu=np.zeros((k,l))
        #print(len(mu))
        sigma=np.zeros((k,l,l))
        prob_z=np.zeros(k)
        for j in range(k):
            for i in range(n):
                prob_z[j]+=w[j,i]
            #print(prob_z)
        prob_z/=n
        for j in range(k):
            for i in range(n):
                mu[j]+=w[j,i]*Xdata[i]
            mu[j]/=w[j,:].sum()
            #print(mu)
        for j in range(k):
            for i in range(n):
                y_s=np.reshape(Xdata[i]-mu[j],(l,1))
                sigma[j]+=w[j,i]*np.dot(y_s,y_s.T)
            sigma[j]/=w[j,:].sum()
            #print(sigma)
        
        lihood_new=0
        for i in range(n):
            s=0
            for j in range(k):
                s+=prob_z[j]*mvn(mu[j],sigma[j]).pdf(Xdata[i])
            lihood_new+=abs(np.log(s))
            if np.abs(lihood_new -lihood_old)<.001:
                break
            lihood_old=lihood_new
    return print(lihood_new,mu)
n = 1000
_mus = np.array([[0,4], [-2,0]])
_sigmas = np.array([[[3, 0], [0, 0.5]], [[1,0],[0,2]]])
_pis = np.array([0.6, 0.4])
xdata = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*n))
                    for pi, mu, sigma in zip(_pis, _mus, _sigmas)])
print(xdata.shape)

# initial guesses for parameters
prob_z = np.random.random(2)
prob_z /= prob_z.sum()
mu = np.random.random((2,2))
sigma = np.array([np.eye(2)] * 2)
z = EM_gmm(xdata, prob_z, mu, sigma)

            
                
                
        
