import numpy
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

# exact solution to Fokker-Planck of a bivariate Brownian motion is a
# bivariate normal distribution diffusing in time in the space (X,Y) .
# + 3D animation

def P(x,y,x0,y0,mx,my,t,sx,sy,r):
    
    """
    returns the probability density of a 2-dimensional Brownian motion.
    parameters:
        x, y : value ranges
        x0, y0 : initial conditions
        mx, my : drift parameters
        sx, sy : diffusion parameters
        r : linear correlation
        t : time t > 0
    """
    
    C = 2*numpy.pi*sx*sy*t*numpy.sqrt(1-r**2)
    
    return (1/C*numpy.exp(-(0.5/(1-r**2))*(
            (x-mx*t-x0)**2/(sx**2*t)
            +(y-my*t-y0)**2/(sy**2*t)
            -2*r*(x-mx*t-x0)*(y-my*t-y0)/(sx*sy*t))))
    
x0 = 1
y0 = 4
mx = 0.03
my = 0.05
sx = 0.15
sy = 0.25
r = -0.3
T = 25

N_x, N_y = (100 for i in range(0,2))
x = numpy.linspace(x0-3,x0+3,N_x)
y = numpy.linspace(y0-3,y0+5,N_y)
dx = numpy.diff(x)[0]
dy = numpy.diff(y)[0]
N_t = 25
t = numpy.linspace(4,T,N_t)

xmesh, ymesh = numpy.meshgrid(x, y)

pdfs = numpy.zeros((N_x,N_y,N_t))
for i in range(0,N_t):
    pdfs[:,:,i] = P(xmesh,ymesh,x0,y0,mx,my,t[i],sx,sy,r)
    print(numpy.sum(pdfs[:,:,i])*dx*dy)
    
def dyn(frame_number, pdfs, im):
    im[0].remove()
    im[0] = ax.plot_surface(xmesh, ymesh, 
            pdfs[:,:,frame_number], cmap=mp.cm.inferno)
    
fig = mp.figure()
ax = fig.add_subplot(111, projection='3d')

im = [ax.plot_surface(xmesh, ymesh, 
                      pdfs[:,:,0], 
                      rstride=1, cstride=1)]
ax.set_zlim(0,0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y,t)')
ani = matplotlib.animation.FuncAnimation(fig, 
                                 dyn, N_t, 
                                 fargs=(pdfs, im), 
                                 interval=20)


mp.show()





