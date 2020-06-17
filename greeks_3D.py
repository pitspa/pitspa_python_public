import numpy
import scipy.stats
import pandas
import matplotlib
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
import math

# closed form partial derivatives of BS pde for call or put options,
# plotted in 3D

St = 100
K = 100
T = 3 # years
rf = 0.02 # yearly
sigma = 0.12 # yearly
div_yield = 0 # yearly
contract_type = 1 # 1: long call, 2: long put, -1: short call, -2: short put

time = numpy.linspace(T,20,num=100)
price = numpy.linspace(1,int(round(St*2,0)),num=100)
price, time = numpy.meshgrid(price, time)

d1_ = (1 /(sigma * numpy.sqrt(time)))*(numpy.log( price / K ) + (rf - div_yield + 0.5*(sigma**2)) * time)
d2_ = d1_ - (sigma * numpy.sqrt(time))

print(numpy.shape(d1_))


if contract_type == 1:

    option = price *(numpy.exp(-div_yield*time)) * scipy.stats.norm(0, 1).cdf(d1_) - K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(d2_)
    delta = numpy.exp(-div_yield*time) * scipy.stats.norm(0, 1).cdf(d1_)
    gamma = numpy.exp(-div_yield*time)*scipy.stats.norm(0, 1).pdf(d1_)*(price*sigma*numpy.sqrt(time))**-1
    theta = -price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*sigma*((2*numpy.sqrt(time))**-1) - rf*K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(d2_) + div_yield*(numpy.exp(-div_yield*time))*St*scipy.stats.norm(0, 1).cdf(d1_)
    vega = price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(time)

    d1_ = (1 /(sigma * math.sqrt(T)))*(numpy.log( St / K ) + (rf  - div_yield + 0.5*(sigma**2)) * T)
    d2_ = d1_ - (sigma * math.sqrt(T))
    print('B-S Value:', St *(numpy.exp(-div_yield*T)) * scipy.stats.norm(0, 1).cdf(d1_) - K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(d2_))
    print('Delta: ',(numpy.exp(-div_yield*T)) * scipy.stats.norm(0, 1).cdf(d1_))
    print('Gamma: ', (numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*(St*sigma*numpy.sqrt(T))**-1)
    print('Theta (one year): ', -St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*sigma*((2*numpy.sqrt(T))**-1) - rf*K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(d2_) + div_yield*(numpy.exp(-div_yield*T))*St*scipy.stats.norm(0, 1).cdf(d1_))
    print('Vega: ', St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(T))

if contract_type == 2:

    option = -price*(numpy.exp(-div_yield*time))* scipy.stats.norm(0, 1).cdf(-d1_) + K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(-d2_)
    delta = -(numpy.exp(-div_yield*time)) * scipy.stats.norm(0, 1).cdf(-d1_)
    gamma = (numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*(price*sigma*numpy.sqrt(time))**-1
    theta = -price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(-d1_)*sigma*((2*numpy.sqrt(time))**-1) + rf*K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(-d2_) - div_yield*(numpy.exp(-div_yield*time))*price*scipy.stats.norm(0, 1).cdf(d1_)
    vega = price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(time)

    d1_ = (1 /(sigma * math.sqrt(T)))*(numpy.log( St / K ) + (rf - div_yield + 0.5*(sigma**2)) * T)
    d2_ = d1_ - (sigma * math.sqrt(T))
    print('B-S Value:', - St *(numpy.exp(-div_yield*T))* scipy.stats.norm(0, 1).cdf(-d1_) + K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(-d2_))
    print('Delta: ',-(numpy.exp(-div_yield*T)) * scipy.stats.norm(0, 1).cdf(-d1_))
    print('Gamma: ', (numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*(St*sigma*numpy.sqrt(T))**-1)
    print('Theta (one year): ', -St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(-d1_)*sigma*((2*numpy.sqrt(T))**-1) + rf*K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(-d2_) - div_yield*(numpy.exp(-div_yield*T))*St*scipy.stats.norm(0, 1).cdf(d1_))
    print('Vega: ', St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(T))
          
if contract_type == -1:

    option = -(price *(numpy.exp(-div_yield*time)) * scipy.stats.norm(0, 1).cdf(d1_) - K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(d2_))
    delta = -((numpy.exp(-div_yield*time)) * scipy.stats.norm(0, 1).cdf(d1_))
    gamma = -((numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*(price*sigma*numpy.sqrt(time))**-1)
    theta = -(-price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*sigma*((2*numpy.sqrt(time))**-1) - rf*K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(d2_) + div_yield*(numpy.exp(-div_yield*time))*price*scipy.stats.norm(0, 1).cdf(d1_))
    vega = -(price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(time))

    d1_ = (1 /(sigma * math.sqrt(T)))*(numpy.log( St / K ) + (rf - div_yield + 0.5*(sigma**2)) * T)
    d2_ = d1_ - (sigma * math.sqrt(T))
    print('B-S Value:', -(St *(numpy.exp(-div_yield*T)) * scipy.stats.norm(0, 1).cdf(d1_) - K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(d2_)))
    print('Delta: ', -((numpy.exp(-div_yield*T)) * scipy.stats.norm(0, 1).cdf(d1_)))
    print('Gamma: ', -((numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*(St*sigma*numpy.sqrt(T))**-1))
    print('Theta (one year): ', -(-St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*sigma*((2*numpy.sqrt(T))**-1) - rf*K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(d2_) + div_yield*(numpy.exp(-div_yield*T))*St*scipy.stats.norm(0, 1).cdf(d1_)))
    print('Vega: ', -(St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(T)))
          
if contract_type == -2:

    option = -(- price *(numpy.exp(-div_yield*time))* scipy.stats.norm(0, 1).cdf(-d1_) + K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(-d2_))
    delta = -(- (numpy.exp(-div_yield*time)) * scipy.stats.norm(0, 1).cdf(-d1_))
    gamma = -((numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*(price*sigma*numpy.sqrt(time))**-1)
    theta = -(-price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(-d1_)*sigma*((2*numpy.sqrt(time))**-1) + rf*K*(numpy.exp(-rf*time))*scipy.stats.norm(0, 1).cdf(-d2_) - div_yield*(numpy.exp(-div_yield*time))*price*scipy.stats.norm(0, 1).cdf(d1_))
    vega = -(price*(numpy.exp(-div_yield*time))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(time))

    d1_ = (1 /(sigma * math.sqrt(T)))*(numpy.log( St / K ) + (rf - div_yield + 0.5*(sigma**2)) * T)
    d2_ = d1_ - (sigma * math.sqrt(T))  
    print('B-S Value:', -(- St *(numpy.exp(-div_yield*T))* scipy.stats.norm(0, 1).cdf(-d1_) + K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(-d2_)))
    print('Delta: ', -(-(numpy.exp(-div_yield*T)) * scipy.stats.norm(0, 1).cdf(-d1_)))
    print('Gamma: ', -((numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*(St*sigma*numpy.sqrt(T))**-1))
    print('Theta (one year): ', -(-St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(-d1_)*sigma*((2*numpy.sqrt(T))**-1) + rf*K*(numpy.exp(-rf*T))*scipy.stats.norm(0, 1).cdf(-d2_) - div_yield*(numpy.exp(-div_yield*T))*St*scipy.stats.norm(0, 1).cdf(d1_)))
    print('Vega: ', -(St*(numpy.exp(-div_yield*T))*scipy.stats.norm(0, 1).pdf(d1_)*numpy.sqrt(T)))


fig = mp.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(price,time,option, cstride=5, rstride=5, cmap=mp.cm.inferno)
ax.set_xlabel('price')
ax.set_zlabel('option_value')
ax.set_ylabel('time_to_maturity')
mp.show()

fig = mp.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(price,time,delta, cstride=2, rstride=2, cmap=mp.cm.inferno)
ax.set_xlabel('price')
ax.set_zlabel('delta')
ax.set_ylabel('time_to_maturity')
mp.show()

fig = mp.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(price,time,gamma, cstride=2, rstride=2, cmap=mp.cm.inferno)
ax.set_xlabel('price')
ax.set_zlabel('gamma')
ax.set_ylabel('time_to_maturity')
mp.show()

fig = mp.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(price,time,theta, cstride=2, rstride=2, cmap=mp.cm.inferno)
ax.set_xlabel('price')
ax.set_zlabel('theta')
ax.set_ylabel('time_to_maturity')
mp.show()

fig = mp.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(price,time,vega, cstride=2, rstride=2, cmap=mp.cm.inferno)
ax.set_xlabel('price')
ax.set_zlabel('vega')
ax.set_ylabel('time_to_maturity')
mp.show()

