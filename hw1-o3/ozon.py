import datetime
from datetime import timedelta
from netCDF4 import Dataset
import numpy as np
import calendar
import matplotlib.pyplot as plt
import json

plt.figure()
cal = calendar.Calendar()

t0 = datetime.date(1979, 1, 1)
nc = Dataset("MSR-2.nc")
lats = nc.variables['latitude'][:]
lons = nc.variables['longitude'][:]
t = nc.variables['time'][:]

lat = 35.68
lon = 135.6

la = np.searchsorted(lats, lat)
lo = np.searchsorted(lons, lon)

newtimes = np.zeros((480,0))
newtimes = np.append(newtimes, t0)

for i in t:
    mn = int((i % 12) + 1)
    y = int(1979 + ((i // 12) - 9))
    b = calendar.monthrange(y, mn)[1]
    j = int(i - 108)
    newtimes = np.append(newtimes, t0 + timedelta(days = b))
    t0 = t0 + datetime.timedelta(days = b)
    if i == 586:
        break

y = nc.variables['Average_O3_column'][:, la, lo]
newtimesjanuary = newtimes[0:480:12]
yjanuary = nc.variables['Average_O3_column'][::12, la, lo]
newtimesjuly = newtimes[6:480:12]
yjuly = nc.variables['Average_O3_column'][6:480:12, la, lo]

minall = min(y)
maxall = max(y)
meanall  = int(sum(y)/len(y))

minjanuary = min(yjanuary )
maxjanuary  = max(yjanuary )
meanjanuary   = int(sum(yjanuary )/len(yjanuary ))

minjuly = min(yjuly)
maxjuly = max(yjuly)
meanjuly  = int(sum(yjuly)/len(yjuly))

plt.plot(newtimes, y, label ='The whole time')
plt.plot(newtimesjanuary, yjanuary, label ='On January')
plt.plot(newtimesjuly, yjuly, label ='On July')

plt.grid()
plt.title("Monthly average of ozone column distribution in Tokyo")
plt.xlabel("Time, months")
plt.ylabel("Ozon contempt, Dobson units")
plt.legend(loc = 'upper left')
plt.savefig('ozon.png')
plt.show()

d ={
  "city": "Tokyo",
  "coordinates": [35.68, 135.6],
  "jan": {
    "min":  283,
    "max":  354,
    "mean": 322
  },
  "jul": {
    "min":  284,
    "max":  320,
    "mean": 300
  },
  "all": {
    "min":  254,
    "max":  388,
    "mean": 311
  }
}
with open('ozon.json', 'w') as write_file:
    json.dump(d, write_file, indent=4, separators=(',', ': '))





