import numpy
import matplotlib.pyplot as mp

# Finite difference scheme for the numerical solution of the SIR epidemic 
# model. Three subpopulations: S: non-infected, I: infected, R: dead/recovered

def St_dt(St,b,N,It,dt):
    
    """

    Parameters
    ----------
    St : initial value of uninfected at time t
    b : infection parameter
    N : total population units
    It : initial value of infected
    dt : time grid interval

    Returns
    -------
    Finite difference scheme S(t+dt) for the uninfected population process 

    """
    
    return St*(1-(b/N)*It*dt)

def It_dt(St,b,g,N,It,dt):
    
    """

    Parameters
    ----------
    St : initial value of uninfected at time t
    b : infection parameter
    g : decay parameter of the infected
    N : total population units
    It : initial value of infected
    dt : time grid interval

    Returns
    -------
    Finite difference scheme I(t+dt) for the infected population process 

    """
    return It*(1+(b/N)*St*dt-g*dt)

def Rt_dt(Rt,It,g,dt):
    
    """

    Parameters
    ----------
    Rt : initial value of recovered/dead at time t
    g : decay parameter of the infected
    N : total population units
    It : initial value of infected
    dt : time grid interval

    Returns
    -------
    Finite difference scheme R(t+dt) for the recovered/dead population process 

    """
    return Rt+g*It*dt

N = 100
b = 20
g = 1.5
R0 = b/g # basic reproduction number

N_t = 20000
T = 3 # terminal time
t = numpy.linspace(0,T,N_t)
dt = numpy.diff(t)[0]

S, I, R, Nt = (numpy.zeros(N_t) for i in range(0,4))
S[0] = N - 1
I[0] = 1 # first infected!
R[0] = 0
Nt[0] = S[0]+ I[0]+ R[0] 
for t_ in range(1,N_t):
    S[t_] = St_dt(S[t_-1],b,N,I[t_-1],dt)
    I[t_] = It_dt(S[t_-1],b,g,N,I[t_-1],dt)
    R[t_] = Rt_dt(R[t_-1],I[t_-1],g,dt)
    Nt[t_] = S[t_] + I[t_] + R[t_] # the total remains constant
    
    
mp.plot(t,S,'b',label='S(t): Uninfected')
mp.plot(t,I,'r',label='I(t): Infected')
mp.plot(t,R,'g',label='R(t): Dead or Recovered')
mp.legend()
mp.xlabel('t')
mp.ylabel('f(t)')
mp.xlim((0,T))
mp.show()
    
    



