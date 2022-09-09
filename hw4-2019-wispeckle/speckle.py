from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from cv2 import resize
from scipy.ndimage import rotate

data = fits.open('speckledata.fits')[2].data

# 2.1

img1 = np.mean(data, axis=0)
img1 = resize(img1, (512, 512))
plt.imsave('mean.png', img1, vmax=500)

# 2.2

img2 = np.fft.ifft2(data)
img2 = np.fft.fftshift(img2)
img2 = np.abs(img2)**2
img2 = np.sum(img2, axis=0)
img20 = img2
img2 = resize(img2, dsize=(512, 512))
plt.imsave('fourier.png', img2, vmax=200)

# 2.3

angles = np.arange(0, 360, 3.6)
img3 = np.zeros((200, 200))
for i in range(100):
    img3 += rotate(img20, angles[i], reshape=False)
img3 = img3/100
img30 = img3
img3 = resize(img3, (512, 512))
plt.imsave('rotaver.png', img3, vmax=200)









