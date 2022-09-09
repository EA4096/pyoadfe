import numpy as np
import math
from scipy.integrate import quad
import opt
import matplotlib.pyplot as plt
import json

x0 = np.array([0.5, 50])


def load():
    global y, mu, z
    y = np.loadtxt('jla_mub.txt', delimiter=' ')
    z = y[:, 0].astype('float64')
    mu = y[:, 1].astype('float64')


c = 3*(10**11)


def mu_t(x):

    def fd(t):
        fd0 = 1 / (math.sqrt((1 - x[0]) * ((1 + t) ** 3) + x[0]))
        return fd0

    m = []
    for i in range(len(z)):
        b = z[i]
        vf = 5 * math.log((c / x[1])*(1 + b) * quad(fd, 0, b)[0], 10) - 5
        m.append(vf)
    return m


def j(x):
    j1 = []
    j2 = []

    def fd(t):
        fd0 = 1 / (math.sqrt((1 - x[0]) * ((1 + t) ** 3) + x[0]))
        return fd0

    def fu(t):
        fu0 = -(5 / (2 * 2.30)) * (((t + 1) ** 3) - 1) / ((x[0] - (x[0] - 1) * ((t + 1) ** 3)) ** (3 / 2))
        return fu0

    for i in range(len(z)):
        b = z[i]
        a0 = quad(fu, 0, b)[0]
        a1 = quad(fd, 0, b)[0]
        p = a0/a1
        j1.append(p)
        j2.append(5/(2.30*x[1]))
    ja = np.zeros((len(z), 2), dtype=float)
    ja[:, 0] = j1
    ja[:, 1] = j2
    return ja


def main():
    global x, result
    load()
    x = x0
    gn = opt.gauss_newton(mu, mu_t, j, x0)
    levberg = opt.lm(mu, mu_t, j, x0)
    return gn, levberg

main()


N1 = [i for i in range(1, (int(opt.gauss_newton(mu, mu_t, j, x0).nfev))+1)]
N2 = [i for i in range(1, (int(opt.lm(mu, mu_t, j, x0).nfev))+1)]



plt.figure()
plt.xlabel('z')
plt.ylabel('mu')
plt.grid()
plt.title('Оптимизация методоv Гаусса - Ньютона', size=10)
plt.plot(z, mu_t(opt.gauss_newton(mu, mu_t, j, x0).x),  color='g', linewidth=2)
plt.scatter(z, mu, color='black')


plt.figure()
plt.xlabel('z')
plt.ylabel('mu')
plt.grid()
plt.title('Оптимизация методом Левенберга - Марквардта', size=10)
plt.plot(z, mu_t(opt.lm(mu, mu_t, j, x0).x),  color='r', linewidth=2)
plt.scatter(z, mu, color='black')
plt.savefig('mu-z.png')


plt.figure()
plt.xlabel('Итерационный шаг', size=12)
plt.ylabel('Сумма потерь', size=12)
plt.grid()
plt.plot(N1, opt.gauss_newton(mu, mu_t, j, x0).cost, label='Для метода Гаусса ньютона')
plt.plot(N2, opt.lm(mu, mu_t, j, x0).cost, label='Для метода Левенберга - Марквардта')
plt.title('Потери', size=15)
plt.legend()
plt.savefig('cost.png')


with open('parametrs.json', 'w') as file:
    json.dump({"Gauss-Newton": {"H0": opt.gauss_newton(mu, mu_t, j, x0).x[1],
                                "Omega": opt.gauss_newton(mu, mu_t, j, x0).x[0],
                                "nfev": opt.gauss_newton(mu, mu_t, j, x0).nfev},
               "Levenberg-Marquardt": {"H0": opt.lm(mu, mu_t, j, x0).x[1],
                                       "Omega": opt.gauss_newton(mu, mu_t, j, x0).x[0],
                                       "nfev": opt.lm(mu, mu_t, j, x0).nfev}},
              file, indent=4, separators=(',', ': '))






