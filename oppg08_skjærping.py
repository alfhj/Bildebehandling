from skimage import io, util
import sys
import scipy
import numpy
from math import pi, exp

def konverter(image, filter):
    sum = numpy.sum(filter)
    filter = filter / (1 if sum == 0 else sum)
    image = scipy.ndimage.convolve(image, filter)
    return numpy.clip(image, 0, 1)

def gauss_filter(size, sigma=None):
    if sigma is None: sigma = (size + 1) / 6
    filter = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            s = i - size // 2 # sentrerte koordinater:
            t = j - size // 2 # midterste pixel er (0,0)
            filter[i, j] = 1 / (2 * pi * sigma ** 2) * exp(-(s ** 2 + t ** 2) / (2 * sigma ** 2))
    return filter

filter1 = numpy.array([
    [0, 1, 0],
    [1, -5, 1],
    [0, 1, 0]
])
filter1a = numpy.array([
    [0, -1, 0],
    [-1, -5, -1],
    [0, -1, 0]
])
filter2 = numpy.array([
    [1, 1, 1],
    [1, -9, 1],
    [1, 1, 1]
])
filter3 = numpy.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])
sobelx = numpy.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobely = numpy.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
# negativt tall i midten gir blur
laplace1 = numpy.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])
laplace2 = numpy.array([
    [-1, 1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

img = util.img_as_float(io.imread(sys.argv[1], as_gray=True))
c = numpy.float(sys.argv[2])
"""
size = numpy.int(sys.argv[2])
if size < 1 or size % 2 == 0:
    print("Størrelsen er på feil format")
    sys.exit(1)

if len(sys.argv) > 3:
    sigma = numpy.float(sys.argv[3])
else: sigma = None

filter0 = gauss_filter(size, sigma)
print(filter0)
print(numpy.sum(filter0))
img = konverter(img, filter0)
"""
#img1 = numpy.abs(konverter(img, sobelx))
#img2 = numpy.abs(konverter(img, sobely))
#img3 = numpy.sqrt(img1 ** 2 + img2 ** 2)# / numpy.sqrt(2)
img4 = scipy.ndimage.convolve(img, laplace1)
img5 = scipy.ndimage.convolve(img, laplace2)
img6 = numpy.clip(img + c * img4, 0, 1)
img7 = numpy.clip(img + c * img5, 0, 1)
img8 = konverter(img, filter1)
img8a = konverter(img, filter1a)
img8 = konverter(img, filter2)
io.imshow(numpy.concatenate([img8, img8a], axis=1))
io.show()