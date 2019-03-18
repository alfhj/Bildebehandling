from skimage import io, util
import sys
import numpy

img = util.img_as_float(io.imread(sys.argv[1], as_gray=True))
bins = numpy.int(sys.argv[2])
hist = [0] * bins
for pixel in numpy.nditer(img):
    bin = numpy.int(pixel * bins)
    hist[bin] += 1

print(hist)