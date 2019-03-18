"""
Program som utfører DFT og iDFT på bilder.
Fungerer kun med gråskala-bilder.
"""
from skimage import exposure, io, util
import numpy

def show_freq(image):
	# fremhever informasjonen i et frekvensbilde for visning
	out = numpy.absolute(image)
	out = numpy.log(1 + out)
	out = exposure.rescale_intensity(out)
	return out

def dft_trans(image):
	# negerer annen hver pixel - dette vil sentrere frekvensplottet
	(h, w) = image.shape
	image = image.copy()
	for i in range(h):
		for j in range(w):
			image[i, j] = image[i, j] * (-1) ** (i + j)
	return image

def dft(image, inverse=False):
	# transformerer et bilde til eller fra frekvensplanet
	(h, w) = image.shape
	sign = 1 if inverse else -1

	out = numpy.zeros((h, w), dtype="complex")
	for u in range(h):
		for v in range(w):
			c_sum = 0
			for i in range(h):
				for j in range(w):
					expo = sign * 2j * numpy.pi * (u * i / h + v * j / w)
					c_sum += image[i, j] * numpy.exp(expo)
			out[u, v] = c_sum
	if inverse:
		out = numpy.absolute(out / (h * w))
	return out

def dft_quick(image, inverse=False):
	# transformerer et bilde til eller fra frekvensplanet
	# denne bruker forskjellige metoder for å gjøre den kjappere
	# baserer seg på eulers formel: e ** (j * x) = cos(x) + j * sin(x)
	(h, w) = image.shape
	sign = 1 if inverse else -1
	expo = numpy.zeros((2, h, w))
	for i in range(h):
		for j in range(w):
			expo[0, i, j] = i / h
			expo[1, i, j] = j / w

	out = numpy.zeros((h, w), dtype="complex")
	for u in range(h):
		for v in range(w):
			expo_cur = sign * 2 * numpy.pi * (u * expo[0] + v * expo[1])
			re = numpy.sum(numpy.cos(expo_cur) * image)
			im = numpy.sum(numpy.sin(expo_cur) * image)
			out[u, v] = complex(re, im)
	if inverse:
		out = numpy.absolute(out / (h * w))
	return out

img = util.img_as_float(io.imread("lena_small.png"))
img = dft_trans(img)
#freq = numpy.fft.fft2(img)
freq = dft_quick(img)
out = dft_quick(freq, inverse=True)
#freq = numpy.fft.ifft2(img)
#out = show_freq(freq)

out = numpy.clip(out, 0, 1)
io.imsave("output.png", util.img_as_ubyte(out))