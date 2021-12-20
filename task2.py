import requests
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import pi, speed_of_light
from scipy.special import spherical_jn as jn
from scipy.special import spherical_yn as yn


def hn(n, x):
    return jn(n, x) + 1j * yn(n, x)


def bn(n, x):
    return (x * jn(n - 1, x) - n * jn(n, x)) / (x * hn(n - 1, x) - n * hn(n, x))


def an(n, x):
    return jn(n, x) / hn(n, x)


link="https://jenyay.net/uploads/Student/Modelling/task_02.csv"
variant=8

content_link=requests.get(link)

content_string=content_link.text

date=content_string.split("\n")[variant].split()

D=float(date[1][:-1])
fmin=float(date[2][:-1])
fmax=float(date[3])
fstep = 1e6

r = D / 2
freq = np.arange(fmin, fmax+fstep, fstep)
lambd = speed_of_light / freq
k = 2 * pi / lambd

arr_sum = [((-1) ** n) * (n + 0.5) * (bn(n, k * r) - an(n, k * r)) for n in range(1, 20)]
summa = np.sum(arr_sum, axis=0)
rcs = (lambd ** 2) / pi * (np.abs(summa) ** 2)

path = pathlib.Path("results")
path.mkdir(exist_ok=True)

jsonfile="{\n    \"data\": [\n"
for i in range(0,len(freq)):
    jsonfile+="        {\"freq\": "+str(freq[i])+", \"lambda\": "+str(lambd[i])+\
    ", \"rcs\": "+str(rcs[i])+"},\n"
    

file = path / "result_task2.json"
out = file.open("w")
out.write(jsonfile)
out.close()

plt.plot(freq / 1e6, rcs)
plt.xlabel("$f, МГц$")
plt.ylabel(r"$\sigma, м^2$")
plt.grid()
plt.savefig("results/task2.png")
plt.show()

