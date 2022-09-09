import numpy as np
import scipy.signal
import json

barker = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
bark = barker*(-1)

data = np.loadtxt('Щекотихин.txt')

# Первый метод


def f(t):
    r = data[55*t:55*t+55]
    s = []
    j = 0
    s0 = 0
    for k in range(11):
        a = r[j:j + 5]
        mean = np.mean(a)
        if mean > 0:
            s = np.append(s, 1)
        else:
            s = np.append(s, -1)
        j = j + 5

    k = 0
    km = 0
    for g in range(11):
        if s[g] == barker[g]:
            k = k + 1
        if s[g] == bark[g]:
            km = km + 1
    if k >= 9:
        s0 = 1
    if km >= 9:
        s0 = 0
    return s0


out = []
for i in range(88):
    out = np.append(out, f(i))

m1 = []
for v in out:
    m1.append(int(v))

d1 = np.asarray(m1, dtype=np.uint8)
byte_array1 = np.packbits(d1)
byte_str1 = byte_array1.tobytes()
s1 = byte_str1.decode(encoding='ascii')

# Второй метод

barker5 = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

convolution = np.convolve(data, barker5)

x = []
for i in convolution:
    x.append(i)

x1 = []
for i in x:
    x1.append(-1*i)

pop = scipy.signal.find_peaks(x, height=40)
nep = scipy.signal.find_peaks(x1, height=40)

ind = np.concatenate((nep[0], pop[0]), axis=0)
ind.sort()

m2 = []
for i in ind:
    if x[i] > 0:
        m2.append(1)
    else:
        m2.append(0)

d2 = np.asarray(m2, dtype=np.uint8)
byte_array2 = np.packbits(d2)
byte_str1 = byte_array2.tobytes()
s2 = byte_str1.decode(encoding='ascii')

print(s1)
print(s2)

with open('wifi.json', 'w') as file:
    json.dump({"message": s2}, file)








