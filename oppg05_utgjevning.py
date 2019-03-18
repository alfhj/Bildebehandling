from skimage import io, util
import sys
import scipy
import numpy
from math import pi, exp

def konverter(image, filter):
    sum = numpy.sum(filter)
    filter = filter / (1 if sum == 0 else sum)
    image = scipy.ndimage.filters.convolve(image, filter)
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
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])
filter2 = numpy.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])
filter3 = numpy.array([
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]
])
filter4 = numpy.array([
    [-1, -1, -1],
    [-1, 7, -1],
    [-1, -1, -1]
])
filter5 = numpy.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
filter6 = numpy.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

img = util.img_as_float(io.imread(sys.argv[1], as_gray=True))

size = numpy.int(sys.argv[2])
if size < 1 or size % 2 == 0:
    print("Størrelsen er på feil format")
    sys.exit(1)

if len(sys.argv) > 3:
    sigma = numpy.float(sys.argv[3])
else: sigma = None

filter0 = gauss_filter(size, sigma)
#print(filter0)
#print(numpy.sum(filter0))
img = konverter(img, filter6)
io.imshow(img)
io.show()