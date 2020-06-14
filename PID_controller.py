import numpy
import matplotlib.pyplot as mp

# PID controller

N = 1000
T = 10
t = numpy.linspace(0,T,N)
dt = numpy.diff(t)[0]

S = numpy.full(shape=N,fill_value=1)

def PID(S,K_p,K_i,K_d,dt):
    
    """
    Returns the PID process that approximates function S(t).
    At every observation of S, the error between the PID and S is 
    calculated. The variation of the PID depends on the current 
    error (proportional control), past terms (integral control),
    and the finite difference between the current error and the 
    precedent error (differential control). The sensitivities must
    be calibrated to ensure convergence.
    parameters:
        S : the function to be approximated by the PID
        K_p, K_i, K_d : proportional, integral and differential sensitivies
        dt : time interval between observations
    
    """
    
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

cont1, err1 = PID(S,0.0125,0.1,0,dt)     
cont2, err2 = PID(S,0.0125,0.1,-0.005,dt) 

mp.plot(t,S,'k--')
mp.plot(t,cont1,'r')
mp.plot(t,cont2,'g')
mp.xlim((t[0],t[-1]))
mp.show()
