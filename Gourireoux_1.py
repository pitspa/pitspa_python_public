import numpy
import scipy.stats
import matplotlib.pyplot as mp

# reproduction of results of
# Inference for Noisy Long Run Component Process C., GOURIEROUX (1) and J., JASIAK

# Figure 1a, 1b, 2
# long term components with varying rho (autocorr. coefficient)
# same noise, unitary variance
# construction of xl(t)(K=1) + xs(t)(K=1000) by using different weights

N = 5000
K = numpy.array([1,100,1000])
rhos = 1-1/K
ars = numpy.zeros((len(K),N)) # xl(t)
e = scipy.stats.norm.rvs(size=N)
ars[:,0] = numpy.full(shape=len(K),fill_value=e[0])
for k in range(0,len(K)):
    for t in range(1,N):
        ars[k,t] = rhos[k]*ars[k,t-1] + numpy.sqrt(1-rhos[k]**2)*e[t]
        
ps = numpy.array([0.95,0.4,0.1]) # len(ps)=len(K) for graph
pl = 1 - ps
ys = numpy.zeros((len(K),N))
for k in range(0,len(K)):
    ys[k,:] = pl[k]*ars[-1,:] + ps[k]*ars[0,:]
     
f_, axarr = mp.subplots(3, 2, sharex=False)
ax1, ax2, ax3, ax4, ax5, ax6 = axarr.flatten()

ax1.plot(ars[0,:],'k',label='K = 1')
ax3.plot(ars[1,:],'r',label='K = 100')
ax5.plot(ars[2,:],'m',label='K = 1000')
ax1.set_xlim((0,N))
ax3.set_xlim((0,N))
ax5.set_xlim((0,N))
ax1.set_ylabel('rl(t)')
ax3.set_ylabel('rl(t)')
ax5.set_ylabel('rl(t)')
ax5.set_xlabel('t')
ax1.legend(loc='lower right')
ax3.legend(loc='upper right')
ax5.legend(loc='upper right')

ax2.plot(ys[0,:],'k',label='ps = 0.95')
ax4.plot(ys[1,:],'r',label='ps = 0.5')
ax6.plot(ys[2,:],'m',label='ps = 0.1')
ax2.set_xlim((0,N))
ax4.set_xlim((0,N))
ax6.set_xlim((0,N))
ax2.set_ylabel('y(t)')
ax4.set_ylabel('y(t)')
ax6.set_ylabel('y(t)')
ax6.set_xlabel('t')
ax2.legend(loc='lower right')
ax4.legend(loc='upper right')
ax6.legend(loc='upper right')

mp.show()
        
# Figure 3, power spectral densities of mixtures

rhos = numpy.array([0.8,0.5,0.3])
fr = numpy.fft.fftfreq(N,1)
pws = numpy.zeros((len(K),N))
for k in range(0,len(K)):
    pws[k,:] = (1-rhos[k]**2)/(1-2*rhos[k]*numpy.cos(2*numpy.pi*fr)+rhos[k]**2)

ps = numpy.array([0.95,0.4,0.1]) # len(ps)=len(K) for graph
pl = 1 - ps 
mixs = numpy.zeros((len(K),N))
for k in range(0,len(K)):
    mixs[k,:] = (ps[k]*(1-rhos[-1]**2)/(1-2*rhos[-1]*numpy.cos(2*numpy.pi*fr)+rhos[-1]**2)
                +pl[k]*(1-rhos[0]**2)/(1-2*rhos[0]*numpy.cos(2*numpy.pi*fr)+rhos[0]**2))

print(numpy.sum(pws[0,:])*(1/N)) # var=1 checked
print(numpy.sum(pws[1,:])*(1/N)) # var=1 checked
print(numpy.sum(pws[2,:])*(1/N)) # var=1 checked

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten()

ax1.plot(fr,pws[0,:],'b.',label='rho = 0.8')
ax1.plot(fr,pws[1,:],'m.',label='rho = 0.5')
ax1.plot(fr,pws[2,:],'r.',label='rho = 0.3')
ax1.legend(loc='upper right')
ax1.set_ylabel('P(f)')
ax1.set_xlabel('f')
ax1.set_xlim((-0.5,0.5))

ax2.plot(fr,mixs[2,:],'y.',label='ps = 0.1')
ax2.plot(fr,mixs[1,:],'g.',label='ps = 0.4')
ax2.plot(fr,mixs[0,:],'k.',label='ps = 0.95')
ax2.legend(loc='upper right')
ax2.set_xlabel('f')
ax2.set_xlim((-0.5,0.5))
mp.show()

# Figure 4
# theoretical autoregressions of running sums
# Formulas in Appendix 2


var = numpy.array([10,20,100,10]) # paired vars and rhos
rhos = numpy.array([0.8,0.9,0.99,0.995])
h = numpy.arange(1,201,1)
Bs = numpy.zeros((len(var),len(h)))
for k in range(0,len(var)):
    for j in range(0,len(h)):
        ghhr = rhos[k]*(1-rhos[k]**h[j])*(1-rhos[k]**h[j])/((1-rhos[k])**2)
        ghr = (1+rhos[k])/(1-rhos[k])*h[j]-2*rhos[k]*(1-rhos[k]**h[j])/((1-rhos[k])**2)
        Bs[k,j] = ghhr/(ghr+h[j]*var[k])

f_, axarr = mp.subplots(2, 2, sharex=False)
ax1, ax2, ax3, ax4 = axarr.flatten()

ax1.plot(h,Bs[0,:],'k',label='rho = 0.8, var = 10')
ax2.plot(h,Bs[1,:],'r',label='rho = 0.9, var = 20')
ax3.plot(h,Bs[2,:],'m',label='rho = 0.99, var = 100')
ax4.plot(h,Bs[3,:],'b',label='rho = 0.995, var = 10')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='lower left')
ax4.legend(loc='lower left')
ax1.set_xlim((1,len(h)))
ax2.set_xlim((1,len(h)))
ax3.set_xlim((1,len(h)))
ax4.set_xlim((1,len(h)))
ax1.set_ylabel('B(h,rho,var)')
ax3.set_ylabel('B(h,rho,var)')
ax3.set_xlabel('h')
ax4.set_xlabel('h')
mp.show()


# Figure 5,6
# Theoretical and estimated coefficients for a (seemingly) wn sequence

N = 400
h = numpy.arange(1,201,1)
rho = 0.99
var = 10
ys = numpy.zeros(N)
xl = numpy.zeros(N)
e1 = scipy.stats.norm.rvs(size=N,loc=0,scale=numpy.sqrt(1-rho**2))
e2 = scipy.stats.norm.rvs(size=N,loc=0,scale=numpy.sqrt(var))
xl[0] = e1[0]
ys[0] = xl[0] + e2[0]
for t in range(1,N):
    # y(t) = xl(t) + xs(t)
    xl[t] = rho*xl[t-1] + e1[t]
    ys[t] = xl[t] + e2[t]
    
mp.plot(ys)
mp.ylabel('y(t)')
mp.xlabel('t')
mp.show()        

ylag = numpy.roll(ys,-1)
ylag2 = numpy.roll(ys,-2)
rhoT = 0.904#(numpy.matmul(ylag2.T,ys))/(numpy.matmul(ylag.T,ys)) # from Eq. 3.1
            # rhoT is hardcoded because of high estimator variance
varT = numpy.matmul(ys.T,ys)/N-1 # from def of y

Bstrue = numpy.zeros(len(h))
BsT = numpy.zeros(len(h))
Bemp = numpy.zeros(len(h))
R2emp = numpy.zeros(len(h))
for j in range(0,len(h)):
    
    ghhr = rho*(1-rho**h[j])*(1-rho**h[j])/((1-rho)**2)
    ghr = (1+rho)/(1-rho)*h[j]-2*rho*(1-rho**h[j])/((1-rho)**2)
    Bstrue[j] = ghhr/(ghr+h[j]*var)
    
    ghhrT = rhoT*(1-rhoT**h[j])*(1-rhoT**h[j])/((1-rhoT)**2)
    ghrT = (1+rhoT)/(1-rhoT)*h[j]-2*rhoT*(1-rhoT**h[j])/((1-rhoT)**2)
    BsT[j] = ghhrT/(ghrT+h[j]*varT)
    
    rsum_ = numpy.zeros(int(N-h[j]))
    for c in range(0,int(N-h[j])):
        rsum_[c] = numpy.sum(ys[int(c):int(c+h[j])])
    rsum_ = rsum_[1:]-numpy.average(rsum_)
    rsumlag = numpy.roll(rsum_,1)
    Bemp[j] = numpy.matmul(rsumlag.T,rsum_)/numpy.matmul(rsumlag.T,rsumlag)
            
mp.plot(h,Bemp,'m',label='Naive autoregression')
mp.plot(h,Bstrue,'k',label='True parameters')
mp.plot(h,BsT,'r',label='Estimated parameters')
mp.legend()
mp.xlim((1,len(h)))
mp.ylabel('B(h,rho,var)')
mp.xlabel('h')
mp.show()

# Table 1a-1e-2

N = 5000
K = numpy.array([1,100,1000])
rhos = 1-1/K
ars = numpy.zeros((len(K),N)) 
e = scipy.stats.norm.rvs(size=N)
ars[:,0] = numpy.full(shape=len(K),fill_value=e[0])
for k in range(0,len(K)):
    for t in range(1,N):
        ars[k,t] = rhos[k]*ars[k,t-1] + numpy.sqrt(1-rhos[k]**2)*e[t]
        
ps = numpy.array([1,0.5,0.95]) 
pl = 1 - ps
ys = numpy.zeros((len(K),N))
for k in range(0,len(K)):
    ys[k,:] = pl[k]*ars[-1,:] + ps[k]*ars[0,:]
    
wn = ys[0,:]
ew = ys[1,:]
ln = ys[2,:]


T = numpy.array([250,500,1000,2000,5000])

means = numpy.zeros((len(K),len(T))) # sample means table
vars_ = numpy.zeros((len(K),len(T))) # sample variances table
q5_ = numpy.zeros((len(K),len(T))) # 5% quantile
q95_ = numpy.zeros((len(K),len(T))) # 95% quantile
Foc = numpy.zeros((len(K),len(T))) # first order correl
rhoest = numpy.zeros((len(K),len(T))) # estimation of rho

for k in range(0,len(K)):
    for t in range(0,len(T)):

        means[k,t] = numpy.average(ys[k,:int(T[t])])
        seq = ys[k,:int(T[t])]
        vars_[k,t] = numpy.matmul(seq.T,seq)/T[t]
        
        sort = numpy.sort(ys[k,:int(T[t])])
        q5_[k,t] = sort[int(0.05*T[t])]
        q95_[k,t] = sort[int(0.95*T[t])]
        
        ylag = numpy.roll(seq,-1)
        Foc[k,t] = (numpy.matmul(ylag.T,seq)/T[t])/vars_[k,t]
        
        ylag2 = numpy.roll(seq,-2)
        rhoest[k,t] = (numpy.matmul(ylag2.T,seq))/(numpy.matmul(ylag.T,seq))
 
meansdf = pandas.DataFrame(means,columns=T,index=['wn','ew','ln']) 
vars_df = pandas.DataFrame(vars_,columns=T,index=['wn','ew','ln'])      
q5_df = pandas.DataFrame(q5_,columns=T,index=['wn','ew','ln'])
q95_df = pandas.DataFrame(q95_,columns=T,index=['wn','ew','ln'])
Focdf = pandas.DataFrame(Foc,columns=T,index=['wn','ew','ln'])
rhoestdf = pandas.DataFrame(rhoest[1:,:],columns=T,index=['ew','ln'])

print('Sample means')
print(meansdf)
print()
     
print('Sample variances')
print(vars_df)
print()
        
print('5% quantile')
print(q5_df)
print()

print('95% quantile')
print(q95_df)
print()

print('First order correlation')
print(Focdf)
print()

print('Rho estimate (Yule-Walker)')
print(rhoestdf)

# Figure 7a-7b

lags = [] # ACFs of equiweighted process
acvs_ew = []
for t in range(0,len(T)):
    seq = ew[:int(T[t])]
    acv = numpy.fft.ifft(numpy.fft.fft(seq)*numpy.conj(numpy.fft.fft(seq)))/int(T[t])
    acvs_ew.append(numpy.real(acv))
    lags.append(numpy.fft.fftfreq(int(T[t]),1/int(T[t])))
    
f_, axarr = mp.subplots(2, 2, sharex=False)
ax1, ax2, ax3, ax4 = axarr.flatten()
    
lmax = 40
ax1.plot(lags[0][:lmax],acvs_ew[0][:lmax]/vars_[1,0],'k.',label='N = 250')
ax2.plot(lags[1][:lmax],acvs_ew[1][:lmax]/vars_[1,1],'k.',label='N = 500')
ax3.plot(lags[2][:lmax],acvs_ew[2][:lmax]/vars_[1,2],'k.',label='N = 1000')
ax4.plot(lags[4][:lmax],acvs_ew[4][:lmax]/vars_[1,4],'k.',label='N = 5000')
ax1.plot(lags[0][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[0])),'r--')
ax2.plot(lags[1][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[1])),'r--')
ax3.plot(lags[2][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[2])),'r--')
ax4.plot(lags[4][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[4])),'r--')
ax1.plot(lags[0][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[0])),'r--')
ax2.plot(lags[1][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[1])),'r--')
ax3.plot(lags[2][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[2])),'r--')
ax4.plot(lags[4][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[4])),'r--')
ax1.set_xlim((-1,lmax-1))
ax2.set_xlim((-1,lmax-1))
ax3.set_xlim((-1,lmax-1))
ax4.set_xlim((-1,lmax-1))
ax1.set_ylim((-0.15,1.1))
ax2.set_ylim((-0.15,1.1))
ax3.set_ylim((-0.15,1.1))
ax4.set_ylim((-0.15,1.1))
ax1.set_ylabel('acf(v)')
ax3.set_ylabel('acf(v)')
ax3.set_xlabel('v')
ax4.set_xlabel('v')
mp.show()

acvs_ln = [] # ACFs of large noise process
for t in range(0,len(T)):
    seq = ln[:int(T[t])]
    acv = numpy.fft.ifft(numpy.fft.fft(seq)*numpy.conj(numpy.fft.fft(seq)))/int(T[t])
    acvs_ln.append(numpy.real(acv))
    
f_, axarr = mp.subplots(2, 2, sharex=False)
ax1, ax2, ax3, ax4 = axarr.flatten()
    
lmax = 40
ax1.plot(lags[0][:lmax],acvs_ln[0][:lmax]/vars_[2,0],'k.',label='N = 250')
ax2.plot(lags[1][:lmax],acvs_ln[1][:lmax]/vars_[2,1],'k.',label='N = 500')
ax3.plot(lags[2][:lmax],acvs_ln[2][:lmax]/vars_[2,2],'k.',label='N = 1000')
ax4.plot(lags[4][:lmax],acvs_ln[4][:lmax]/vars_[2,4],'k.',label='N = 5000')
ax1.plot(lags[0][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[0])),'r--')
ax2.plot(lags[1][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[1])),'r--')
ax3.plot(lags[2][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[2])),'r--')
ax4.plot(lags[4][:lmax],numpy.full(shape=lmax,fill_value=1.96*1/numpy.sqrt(T[4])),'r--')
ax1.plot(lags[0][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[0])),'r--')
ax2.plot(lags[1][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[1])),'r--')
ax3.plot(lags[2][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[2])),'r--')
ax4.plot(lags[4][:lmax],numpy.full(shape=lmax,fill_value=-1.96*1/numpy.sqrt(T[4])),'r--')
ax1.set_xlim((-1,lmax-1))
ax2.set_xlim((-1,lmax-1))
ax3.set_xlim((-1,lmax-1))
ax4.set_xlim((-1,lmax-1))
ax1.set_ylim((-0.15,1.1))
ax2.set_ylim((-0.15,1.1))
ax3.set_ylim((-0.15,1.1))
ax4.set_ylim((-0.15,1.1))
ax1.set_ylabel('acf(v)')
ax3.set_ylabel('acf(v)')
ax3.set_xlabel('v')
ax4.set_xlabel('v')
mp.show()

# Figure 8a-8b
# functional estimates of the distributions

ew2k = ew[:2000]
ew5k = ew[:5000]

U = 300
rhoT = 0.999 # assume known
umax = 3
umin = -umax
u = numpy.linspace(umin,umax,U)
Dpsi_urho_v = numpy.zeros(U,dtype=numpy.complex) # dbs(u)/du (noise part)
Dpsi_urho_v5k = numpy.zeros(U,dtype=numpy.complex)
for a in range(0,U):
    
    ew2klag = numpy.roll(ew2k,-1)
    ew5klag = numpy.roll(ew5k,-1)
    
    v = -u[a]/rhoT
    
    Dpsi_urho_v[a] = ((-numpy.sum(ew2klag*numpy.sin(u[a]*ew2k+v*ew2klag))+1j*numpy.sum(ew2klag*numpy.cos(u[a]*ew2k+v*ew2klag)))/
                        (numpy.sum(numpy.cos(u[a]*ew2k+v*ew2klag))+1j*numpy.sum(numpy.sin(u[a]*ew2k+v*ew2klag))))
    
    Dpsi_urho_v5k[a] = ((-numpy.sum(ew5klag*numpy.sin(u[a]*ew5k+v*ew5klag))+1j*numpy.sum(ew5klag*numpy.cos(u[a]*ew5k+v*ew5klag)))/
                        (numpy.sum(numpy.cos(u[a]*ew5k+v*ew5klag))+1j*numpy.sum(numpy.sin(u[a]*ew5k+v*ew5klag))))

Dpsi_0_v = numpy.zeros(U,dtype=numpy.complex) # dcl(u)/du (x part)
Dpsi_0_v5k = numpy.zeros(U,dtype=numpy.complex)
for a in range(0,U):
    
    ew2klag = numpy.roll(ew2k,-1)
    ew5klag = numpy.roll(ew5k,-1)
    
    v = 0
    Dpsi_0_v[a] = ((-numpy.sum(ew2klag*numpy.sin(u[a]*ew2k+v*ew2klag))+1j*numpy.sum(ew2klag*numpy.cos(u[a]*ew2k+v*ew2klag)))/
                        (numpy.sum(numpy.cos(u[a]*ew2k+v*ew2klag))+1j*numpy.sum(numpy.sin(u[a]*ew2k+v*ew2klag))))

    Dpsi_0_v5k[a] = ((-numpy.sum(ew5klag*numpy.sin(u[a]*ew5k+v*ew5klag))+1j*numpy.sum(ew5klag*numpy.cos(u[a]*ew5k+v*ew5klag)))/
                        (numpy.sum(numpy.cos(u[a]*ew5k+v*ew5klag))+1j*numpy.sum(numpy.sin(u[a]*ew5k+v*ew5klag))))

dclu_du = Dpsi_urho_v - Dpsi_0_v
dclu_du5k = Dpsi_urho_v5k - Dpsi_0_v5k

f_, axarr = mp.subplots(2, 2, sharex=False)
ax1, ax2, ax3, ax4 = axarr.flatten()

ax1.plot(u,numpy.real(Dpsi_urho_v),'k--')
ax3.plot(u,numpy.imag(Dpsi_urho_v),'r--')
ax1.plot(u,numpy.real(Dpsi_urho_v5k),'k-')
ax3.plot(u,numpy.imag(Dpsi_urho_v5k),'r-')
ax2.plot(u,numpy.real(dclu_du),'k--')
ax4.plot(u,numpy.imag(dclu_du),'r--')
ax2.plot(u,numpy.real(dclu_du5k),'k-')
ax4.plot(u,numpy.imag(dclu_du5k),'r-')
ax1.set_xlim((umin,umax))
ax2.set_xlim((umin,umax))
ax3.set_xlim((umin,umax))
ax4.set_xlim((umin,umax))
ax1.set_ylabel('Real part')
ax3.set_ylabel('Imaginary part')
ax3.set_xlabel('u')
ax4.set_xlabel('u')
ax1.set_title('noise comp.')
ax2.set_title('x comp.')
mp.show()
    
    











        









        
    
    