import numpy
import matplotlib.pyplot as mp
import statsmodels

# decomposition of a discrete time series usin the Haar wavelet

def y(N,a,var):
    return  statsmodels.tsa.arima_process.arma_generate_sample(
            ar=[1,-a],
            ma=[1],
            nsample=N,
            sigma=numpy.sqrt(var)
            )

def Psi(N): # returns non-orthogonal Haar wavelet matrix
    H = numpy.array([[1,1],
                      [1,-1]])
    
    r = numpy.arange(1,int(numpy.log2(N)),1)
    for h in r:
        H = numpy.vstack((
                numpy.kron(H,numpy.array([1,1])),
                numpy.kron(numpy.eye(2**h),[1,-1]))  
                )
    return H

def Haar(N): # returns the Haar transform matrix
    H = numpy.array([[1,1],
                      [1,-1]])
    
    r = numpy.arange(1,int(numpy.log2(N)),1)
    for h in r:
        H = numpy.vstack((
                numpy.kron(H,numpy.array([1,1])),
                numpy.kron(numpy.eye(2**h),[1,-1]))  
                )
    r = numpy.arange(1,int(numpy.log2(N))+1,1)
    for h in r:
        H[int(2**(h-1)):int(2**(h)),:] =  H[int(2**(h-1)):int(2**(h)),:]*2**((int(h)-1)/2)
    
    return H

eg = Psi(2**3)
ortheg = numpy.matmul(eg.T,eg)

eg2 = Haar(2**5)

f_, axarr = mp.subplots(1, 3, sharex=False)
ax1, ax2, ax3 = axarr.flatten()


ax1.plot(eg2[1,:],drawstyle='steps-post')
ax1.set_ylabel('psi(t)')
ax1.set_xlabel('t')
ax1.set_ylim((-2.25,2.25))
ax1.set_xlim((0,2**5-1))
ax1.plot(numpy.full(shape=2**5,fill_value=0),'k')


ax2.plot(eg2[2,:],drawstyle='steps-post')
ax2.plot(eg2[3,:],drawstyle='steps-post')
ax2.set_xlabel('t')
ax2.set_ylim((-2.25,2.25))
ax2.set_xlim((0,2**5-1))
ax2.plot(numpy.full(shape=2**5,fill_value=0),'k')


ax3.plot(eg2[4,:],drawstyle='steps-post')
ax3.plot(eg2[5,:],drawstyle='steps-post')
ax3.plot(eg2[6,:],drawstyle='steps-post')
ax3.plot(eg2[7,:],drawstyle='steps-post')
ax3.set_xlabel('t')
ax3.set_ylim((-2.25,2.25))
ax3.set_xlim((0,2**5-1))
ax3.plot(numpy.full(shape=2**5,fill_value=0),'k')

mp.show()

#

N = 2**8
a = 0
s = 0.015 
t = numpy.arange(0,N,1)

sin = numpy.cos(2*numpy.pi*t*4/N)
sin[:int(N/3)] = sin[-int(N/3):] = 0
rt = y(N,a,s) + sin

M = Haar(N)
orth = numpy.matmul(M.T,M)/N

Hr = numpy.matmul(M,rt)/numpy.sqrt(N)

f_, axarr = mp.subplots(1, 3, sharex=False)
ax1, ax2, ax3 = axarr.flatten()

ax1.plot(rt,'k')
ax1.set_xlim((0,N))
ax1.set_ylabel('rt')
ax1.set_xlabel('t')

q = numpy.arange(0,N,1)

for h in numpy.arange(1,int(numpy.log2(N)),1):
    ax2.axvline(x=numpy.log2(2**h),ymin=-5000,ymax=50000,c='black',linestyle='--')
ax2.set_xlim((0,numpy.log2(N)))
ax2.plot(numpy.log2(q),Hr,'b')
ax2.set_ylabel('h(k)')
ax2.set_xlabel('log2(k)')

print(numpy.linalg.norm(rt-numpy.matmul(M.T,Hr)/numpy.sqrt(N))**2)

# pure wavelet decomposition

q = 3 # from q to q+1
cutoff = 2**q
Mq = Haar(N)
Mq[:cutoff,:] = Mq[int(2**(q+1)):,:] = 0

rW = numpy.matmul(Mq.T,Hr)/numpy.sqrt(N)

ax3.plot(rt,'k')
ax3.plot(rW,'r')
ax3.set_ylabel('rt')
ax3.set_xlabel('t')
ax3.set_xlim((0,N))
ax3.legend(('rt','psi(3)'))
mp.show()


        
    