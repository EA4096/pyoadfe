import numpy as np
from scipy.optimize import minimize


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):

    def e_step(p, t, m1, s1, m2, s2):
        t_n1 = t / np.sqrt(2 * np.pi * s1) * np.exp(-0.5 * (p - m1) ** 2 / s1)
        t_n2 = (1 - t) / np.sqrt(2 * np.pi * s2) * np.exp(-0.5 * (p - m2) ** 2 / s2)
        return np.vstack((t_n1 / (t_n1 + t_n2), t_n2 / (t_n1 + t_n2)))

    def m_step(p, old):
        global tau, mu1, sigma1, mu2, sigma2
        t_n1, t_n2 = e_step(p, *old)
        tau = np.sum(t_n1) / (np.sum(t_n1 + t_n2))
        mu1 = np.sum(t_n1 * p) / np.sum(t_n1)
        mu2 = np.sum(t_n2 * p) / np.sum(t_n2)
        sigma1 = np.sum(t_n1 * (p - mu1) ** 2) / np.sum(t_n1)
        sigma2 = np.sum(t_n2 * (p - mu2) ** 2) / np.sum(t_n2)
        return tau, mu1, sigma1, mu2, sigma2

    th = [tau, mu1, sigma1, mu2, sigma2]

    while True:
        th = m_step(x, th)
        if np.linalg.norm(np.array(m_step(x, th)) - th)/np.linalg.norm(th) < rtol:
            break
    res = tuple((th[0], th[1], np.sqrt(th[2]), th[3], np.sqrt(th[4])))
    return res


def max_likelihood(x, tm, mu1m, sigma1m, mu2m, sigma2m, rtol=1e-3):
    x = np.asarray(x)
    xi = np.asarray([tm, mu1m, sigma1m, mu2m, sigma2m])

    def like(t):
        p1 = t[0] / np.sqrt(2 * np.pi * t[2]**2) * np.exp(-0.5 * (x - t[1]) ** 2 / t[2]**2)
        p2 = (1 - t[0]) / np.sqrt(2 * np.pi * t[4]**2) * np.exp(-0.5 * (x - t[3]) ** 2 / t[4]**2)
        s = np.sum(np.log(p1 + p2))
        return -s

    results = minimize(like, x0=xi, method='Nelder-Mead',  tol=rtol)
    res = tuple((results.x[0], results.x[1], results.x[2], results.x[3], results.x[4]))
    return res


def em_double_cluster(x, uniform_dens, tau1, mu1, sigma1, tau2, mu2, sigma2, rtol=1e-3):

    # Делим двумерное распределение на два одномерных и находим параметры каждого из них

    y = x[:, 0]
    z = x[:, 1]

    def e_step2(t, tau1d, mu1d, sigma1d, tau2d, mu2d, sigma2d):
        t_n1 = tau1d / np.sqrt(2 * np.pi * sigma1d) * np.exp(-0.5 * (t - mu1d) ** 2 / sigma1d)
        t_n2 = tau2d / np.sqrt(2 * np.pi * sigma2d) * np.exp(-0.5 * (t - mu2d) ** 2 / sigma2d)
        idt = (t > -0.75) & (t < 0.75)     # Константы - для накождения центров скоплений
        t_u = np.zeros_like(t)
        t_u[idt] = (1 - tau1d - tau2d) * uniform_dens
        return np.vstack((t_n1 / (t_n1 + t_n2 + t_u), t_n2 / (t_n1 + t_n2 + t_u), t_u / (t_n1 + t_n2 + t_u)))

    def m_step2(s, old):
        t_n1, t_n2, t_u = e_step2(s, *old)
        t1 = np.sum(t_n1) / np.sum(t_n1 + t_n2 + t_u)
        t2 = np.sum(t_n2) / np.sum(t_n1 + t_n2 + t_u)
        m1 = np.sum(t_n1 * s) / np.sum(t_n1)
        m2 = np.sum(t_n2 * s) / np.sum(t_n2)
        s1 = np.sum(t_n1 * (s - m1) ** 2) / np.sum(t_n1)
        s2 = np.sum(t_n2 * (s - m2) ** 2) / np.sum(t_n2)
        return t1, m1, s1, t2, m2, s2

    thy = [tau1, mu1[0], sigma1, tau2, mu2[0], sigma2]

    while True:
        thy = m_step2(y, thy)
        if np.linalg.norm(np.array(m_step2(y, thy)) - thy) / np.linalg.norm(thy) < rtol:
            break

    thz = [tau1, mu1[1], sigma1, tau2, mu2[1], sigma2]

    while True:
        thz = m_step2(z, thz)
        if np.linalg.norm(np.array(m_step2(z, thz)) - thz) / np.linalg.norm(thz) < rtol:
            break

    return thz[0], [thy[1], thz[1]], np.sqrt(thy[2]), thz[3], [thy[4], thz[4]], np.sqrt(thy[5])
























