import numpy
import scipy.stats
import matplotlib.pyplot as mp
import matplotlib.animation

# a solution to the wave equation using Fourier transforms

def psi_0x(x):
    return (numpy.sqrt((scipy.stats.norm.pdf(x,loc=0,scale=0.5)+
                      scipy.stats.norm.pdf(x,loc=-1.5,scale=0.75))/2)
                        )*numpy.exp(2j*x)
    
N_x = 1000
x = numpy.linspace(-6,6,N_x)
dx = numpy.diff(x)[0]

N_k = 800
k = numpy.linspace(-20,20,N_k)
dk = numpy.diff(k)[0]

xmesh, kmesh = numpy.meshgrid(x, k)
G = numpy.exp(-1j*kmesh*xmesh) # fourier transform matrix

phi_0k = numpy.matmul(G,psi_0x(x))*dx
  
N_t = 30
T = 2
c = 4.5 # x-displacement parameter
t = numpy.linspace(0,T,N_t)

fig2 = mp.figure() 
ims = []

# wave displacement in two different directions

psi_tx = numpy.zeros((N_t,N_x),dtype=numpy.complex)
psi_txopp = numpy.zeros((N_t,N_x),dtype=numpy.complex)
for i in range(0,N_t):
    psi_tx[i,:] = numpy.matmul(numpy.conj(G.T),phi_0k*numpy.exp(-1j*k*c*t[i]))*dk
    psi_txopp[i,:] = numpy.matmul(numpy.conj(G.T),phi_0k*numpy.exp(-1j*k*(-c)*t[i]))*dk
    gee = mp.plot(x,numpy.imag(psi_tx[i,:]),'r--',
                  x,numpy.imag(psi_txopp[i,:]),'k--',
                  x,numpy.real(psi_tx[i,:]),'r',
                  x,numpy.real(psi_txopp[i,:]),'k',
                  )
    ims.append(gee)
    
im_ani = matplotlib.animation.ArtistAnimation(
        fig2, ims, interval=80, repeat=True, blit=True)

mp.xlim((x[0],x[-1]))
mp.show()


