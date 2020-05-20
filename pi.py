import numpy
import matplotlib
import matplotlib.pyplot as mp

# estimate pi with montecarlo darts

in_circle = 0
out_circle = 0
in_circle_xs = []
in_circle_ys = []
out_circle_xs = []
out_circle_ys = []

for t in range(1, 10000):
    x = 2 * numpy.random.random_sample() - 1
    y = 2 * numpy.random.random_sample() - 1
    if x ** 2 + y ** 2 <= 1:
        in_circle += 1
        in_circle_xs.append(x)
        in_circle_ys.append(y)
    else:
        out_circle += 1
        out_circle_xs.append(x)
        out_circle_ys.append(y)
        
in_square = in_circle + out_circle
circle_square_ratio = in_circle / in_square
square_area = 4
pi = circle_square_ratio * square_area

# A_q : square area
# A : circle area
# A_q = (2r) ** 2 = 4r ** 2
# A / A_q = A / (4r ** 2)
# pi = A / (r ** 2)
# pi = [A / (4r ** 2)] * 4 = (A / A_q) * 4

print(pi)

mp.plot(in_circle_xs, in_circle_ys, 'b.')
mp.plot(out_circle_xs, out_circle_ys, 'r.')
mp.grid(True)
mp.show()


