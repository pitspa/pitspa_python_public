import numpy
import matplotlib.pyplot as mp

# the PID controller will approximate the function S by adjusting the error

N = 1000
T = 10
t = numpy.linspace(0,T,N)
dt = T/N

S = numpy.full(shape=N,fill_value=1)

def PID(S,K_p,K_i,K_d):
    PID_ = numpy.zeros(len(S))
    e_t = numpy.zeros(len(S))
    e_t[0] = S[0] - PID_[0]
    for i in range(1,len(S)):
        e_t[i] = S[i] - PID_[i-1]
        PID_[i] = (PID_[i-1] + 
                    K_p*e_t[i] + # Proportional
                    K_i*numpy.sum(e_t[:i])*dt + # Integral
                    K_d*(e_t[i] - e_t[i-1])/dt) # Differential
    return PID_, e_t

cont1, err1 = PID(S,0.0125,0.1,0)     
cont2, err2 = PID(S,0.0125,0.1,-0.005) 

mp.plot(t,S,'b')
mp.plot(t,cont1,'r')
mp.plot(t,cont2,'g')
mp.grid(True)
mp.show()

