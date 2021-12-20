import pylab
import numpy as np
import pathlib

A = 9.66459


def f(x):
    return -np.abs(np.sin(x)*np.cos(A)*np.exp(np.abs(1-
    np.sqrt(x**2+A**2)/np.pi)))


x_min = -10
x_max = 10
dx = 0.001

x = np.arange(x_min, x_max+dx, dx)
y = f(x)


path = pathlib.Path("results")
path.mkdir(exist_ok=True)
file = path / "result_task1.csv"

csvfile="N, x, y \n"

for i in range(0, len(x)):
    csvfile+= str(i)+", " +str(x[i])+ ', '  +str(y[i])+"\n"
    
out = file.open("w")
out.write(csvfile)
out.close()


pylab.plot(x, y,lw=2)
pylab.grid()
pylab.savefig("results/task1.png")
pylab.show()
