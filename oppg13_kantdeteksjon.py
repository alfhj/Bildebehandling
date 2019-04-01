from math import isclose
from skimage import io, util, morphology
import numpy
import scipy

sobelx = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobely = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
laplace1 = numpy.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
laplace2 = numpy.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

def convolve(image, filter):
	sum = numpy.sum(filter)
	filter = filter / (1 if abs(sum) < 1e-10 else sum)
	out = scipy.ndimage.filters.convolve(image, filter)
	#out = numpy.clip(out, 0, 1)
	return out

def gauss_filter(w, sigma):
	(y, x) = numpy.mgrid[:w, :w]
	dist2 = (y - w // 2) ** 2 + (x - w // 2) ** 2
	out = numpy.exp(-dist2 / (2 * sigma ** 2))
	return out
	
def laplace_gauss_filter(w, sigma):
	(y, x) = numpy.mgrid[:w, :w]
	dist2 = (y - w // 2) ** 2 + (x - w // 2) ** 2
	del2 = (dist2 - 2 * sigma ** 2) / (sigma ** 4)
	out = del2 * numpy.exp(-dist2 / (2 * sigma ** 2))
	out -= numpy.sum(out) / w ** 2
	return out

img = util.img_as_float(io.imread("bilder/lena.tif", as_gray=True))
(h, w) = img.shape
#gauss = gauss_filter(7, 3.0)
#blur = convolve(img, gauss)
#marr1 = convolve(blur, laplace2)
lgauss = laplace_gauss_filter(9, 2.0)
marr2 = convolve(img, lgauss)

pad = numpy.pad(marr2, 1, mode="constant", constant_values=0)
out = numpy.zeros((h, w), dtype=float)
for i in range(h):
	for j in range(w):
		diag = [
			(pad[i+0, j+0], pad[i+2, j+2]),
			(pad[i+0, j+1], pad[i+2, j+1]),
			(pad[i+0, j+2], pad[i+2, j+0]),
			(pad[i+1, j+0], pad[i+1, j+2])
		]
		vals = [p[0] - p[1] for p in diag if p[0] * p[1] < 0]
		# henter største differanse mellom diagonale piklser med forskjellig fortegn
		if vals: out[i, j] = max(vals)

out = out > numpy.mean(out) * 5
out = (out * 255).astype(numpy.uint8)
io.imsave("output.png", out)