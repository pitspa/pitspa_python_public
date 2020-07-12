import numpy
import scipy.stats
import matplotlib.pyplot as mp

# Function approximation on the unit interval using ANN

# simple FF neural network:
# one input in R, one output in R, one hidden layer of N neurons.
# each hidden neuron is linked to both input and output.



def act(x):
    
    """
    activ. function / output function.
    act(x) is sigmoidal, then act(x) is discriminatory
    and it satisfies the ANN universal approximation theorem
    (Cybenko, 1989).  
    
    """
    
    return 1/(1 + numpy.exp(-x))

def dact_dx(y): 
    
    """
    derivative of activ. function;
    argument is the function itself.
       
    """
    
    return y*(1-y)

def D_w_outer(input_,output_,true_,alpha): 
                                           
    """
    Delta of weights connecting hidden layer to output value.
    Parameters:
        input_: output of the hidden layer neuron
        output_: output of the ANN
        true_: true function value
        alpha: learning parameter
    
    """
    
    dE_dw = input_*dact_dx(output_)*(output_ - true_)
    
    return -alpha*dE_dw

def D_w_inner(input_,output_,w_outer,output_outer,true_,alpha):
    
    """
    Delta of weights connecting input value to hidden layer.
    Parameters:
        input_: input variable x
        output_: output of the hidden layer neuron
        w_outer: corresponding weight connecting the 
                 neuron output to the output function of the ANN
        output_outer: output of the ANN
        alpha: learning parameter
    
    """
    
    dE_do = w_outer*dact_dx(output_outer)*(output_outer - true_)
    dE_dw = input_*dact_dx(output_)*dE_do
    
    return -alpha*dE_dw

# random initialization of weight parameters:
    
w = scipy.stats.uniform.rvs(size=4) # weights
b = scipy.stats.uniform.rvs(size=int(len(w)/2)) # constants

                      # even number of total weights ; len(w)/2 is 
                      # number of neurons in hidden layer and number of const.
                      
alpha = 6 # learning parameter

N_x = 500 # learning depends also on the grid spacing
x = numpy.linspace(0,1,N_x)
fx = numpy.sin(2*numpy.pi*(x))**6*numpy.exp(-x**2)
# function to be approximated (bound bw [0,1])

I = 4 # number of learning iterations
output_ = numpy.zeros((I,N_x))
for v in range(0,I): # backpropagation / gradient descent

    for x_ in range(0,N_x):
        
        input_ = x[x_]
        
        neur = numpy.array([])
        for i in range(0,int(len(w)/2)):
            neur = numpy.append(neur,act(input_*w[i]+b[i]))
            
        output_[v,x_] = act(numpy.matmul(neur.T,w[int(len(w)/2):]))

        for j in range(int(len(w)/2),len(w)):
            w[j] += D_w_outer(neur[int(j-int(len(w)/2))],output_[v,x_],fx[x_],alpha)
     
        for i in range(0,int(len(w)/2)):
            w[i] += D_w_inner(input_,neur[i],w[int(len(w)/2+i)],output_[v,x_],fx[x_],alpha)
            b[i] += D_w_inner(1,neur[i],w[int(len(w)/2+i)],output_[v,x_],fx[x_],alpha)
     
mp.plot(x,fx,'k--',label='true f(x)')
mp.plot(x,output_[:,:].T,'.')
mp.xlim((x[0],x[-1]))
mp.ylabel('f(x)')
mp.xlabel('x')
mp.legend()
mp.show()
        
        