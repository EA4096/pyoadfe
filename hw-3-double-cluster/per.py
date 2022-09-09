import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import numpy as np
import matplotlib.pyplot as plt
import mixfit
import json

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    column_filters={'Bmag': '<13'},  # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.5 * u.deg,
    height=1.5 * u.deg,
    catalog='USNO-A2.0',
)[0]
ra = stars['RAJ2000']._data   # прямое восхождение, аналог долготы
dec = stars['DEJ2000']._data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi)
x2 = dec - dec.mean()              # Нормировка
x = np.vstack((x1, x2)).T

res = mixfit.em_double_cluster(x, 2/3, 0.2, [-0.3, 0], 0.01, 0.6, [0.3, 0], 0.01)
plt.hist2d(*x.T, bins=20)

plt.plot(*x.T, '.', color='yellow')

# Центры скоплений и картинка

plt.scatter(res[1][0], res[1][1], marker='x', s=5000, color='red')
plt.scatter(res[4][0], res[4][1], marker='x', s=5000, color='red')
plt.grid(res[1][0], 'Центры скоплений')
plt.xlabel('Прямое восхождение (отнормированное)')
plt.ylabel('Склонение (отнормированное)')
plt.title('Рассеяние точек звёздного поля')
plt.savefig('per.png')
plt.show()


a = res[1][1] + dec.mean()
b = res[4][1] + dec.mean()
c = res[1][0] + res[1][0] / np.cos(res[1][1] / 180 * np.pi) + ra.mean()
d = res[4][0] + res[4][0] / np.cos(res[1][1] / 180 * np.pi) + ra.mean()

with open('per.json', 'w') as file:
    json.dump({
        "size_ratio": 1.5,
        "clusters": [
            {
                "center": {"ra": c, "dec": a},
                "sigma": res[2],
                "tau": res[0]
            },
            {
                "center": {"ra": d, "dec": b},
                "sigma": res[5],
                "tau": res[3]
            }
        ]
        },
        file,
        indent=4,
        separators=(',', ': '))


