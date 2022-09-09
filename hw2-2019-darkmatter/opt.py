#!/usr/bin/env python3
import numpy as np
from collections import namedtuple
from numpy.linalg import inv


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))


def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    global x
    x = x0
    n = 0
    cost = []
    while True:
        n = n + 1  # Считаем шаги
        r = y - f(x)
        cost.append(0.5 * np.dot(r, r))  # Значение функции ошибок
        a0 = j(x)
        jt = a0.transpose()
        a1 = np.dot(jt, a0)
        a2 = inv(a1)
        a3 = np.dot(a2, jt)
        a4 = np.dot(a3, r)
        gradn = np.dot(a0.transpose(), r)
        if np.linalg.norm(a4) <= tol:
            break
        x = x - k * a4
    cost = np.array(cost)
    return Result(nfev=n, cost=cost, gradnorm=np.linalg.norm(gradn), x=x)


def lm(y, f, j, x0, lmbd0=1e-2, v=2, tol=1e-4):
    global x
    x = x0
    n = 0
    cost = []
    i = np.eye(2)

    def dx(lmbd0, jac, r):
        a1 = np.dot(jac.T, jac)
        a2 = a1 + i*lmbd0
        a3 = inv(a2)
        a4 = np.dot(a3, jac.T)
        a5 = np.dot(a4, r)
        return a5

    while True:

        n = n + 1
        r = y - f(x)

        cost.append(0.5 * np.dot(r, r))
        jac = j(x)
        gradl = np.dot(jac.T, r)
        fi = 0.5*np.dot(r, r)

        dx1 = dx(lmbd0, jac, r)
        x = x - dx1
        r1 = y - f(x)
        fi1 = 0.5 * np.dot(r1, r1)
        x = x + dx1

        dx2 = dx(lmbd0/v, jac, r)
        x = x - dx2
        r2 = y - f(x)
        fi2 = 0.5 * np.dot(r2, r2)
        x = x + dx2

        if fi1 <= fi:
            lmbd0 = lmbd0/v

        if (fi2 > fi) and (fi2 < fi1):
            lmbd0 = lmbd0

        if (fi2 > fi) and (fi2 > fi1):
            fi3 = 0
            while fi3 <= fi:
                lmbd0 = lmbd0 * v
                dx3 = dx(lmbd0, jac, r)
                x = x - dx3
                r3 = y - f(x)
                fi3 = 0.5 * np.dot(r3, r3)
                print(fi3)
                x = x + dx3

        if np.linalg.norm(dx(lmbd0, jac, r)) <= tol:
            break

        x = x - dx(lmbd0, jac, r)

    cost = np.array(cost)
    return Result(nfev=n, cost=cost, gradnorm=np.linalg.norm(gradl), x=x)














