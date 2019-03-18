"""
Program som filtrerer bilder i frekvensdomenet.
Fungerer kun med gråskala-bilder.
"""
from skimage import exposure, io, util
import numpy

PF = 2.0 # pad faktor - hvor stor paddingen skal være

def show_freq(image):
	# fremhever informasjonen i et frekvensbilde for visning
	out = numpy.absolute(image)
	out = numpy.log(1 + out)
	out = exposure.rescale_intensity(out)
	return out

def zero_pad(image, out_shape):
	# zero-padder et bilde på høyre og nedre kant
	#  slik at det blir så stort som out_shape
	(h, w) = image.shape
	(ho, wo) = out_shape
	top = (ho - h) // 2
	left = (wo - w) // 2
	out = numpy.zeros(out_shape)
	out[:h, :w] = image
	return out

def dft_trans(image):
	# negerer annen hver pixel - dette vil sentrere frekvensplottet
	(h, w) = image.shape
	out = image.copy()
	for i in range(h):
		for j in range(w):
			out[i, j] = image[i, j] * (-1) ** (i + j)
	return out

def hard(d, d0, **kwargs):
	# ideelt lavpassfilter - frekvensen er gitt av d0
	# 1 hvis avstanden fra sentrum er innenfor d0, 0 ellers
	return (d <= d0).astype(float)

def soft(d, d0, **kwargs):
	# lineært lavpassfilter
	# d0 beskriver området som skal være innenfor 50% lys
	return 1 - numpy.clip(d, 0, d0 * 2) / (d0 * 2)
	
def gauss(d, d0, **kwargs):
	# normalfordelt lavpassfilter
	# d0: bredden på gauss-filteret, innenfor 60.7% lys
	return numpy.exp(-d ** 2 / (2 * d0 ** 2))

def butterworth(d, d0, n=2, **kwargs):
	# butterworthfilter - mellomting av gaussian og ideelt filter
	return 1 / (1 + (d / d0) ** (2 * n))

def laplace(d, **kwargs):
	# laplacefilter
	return -4 * numpy.pi ** 2 * d ** 2

def radial_filter(function, shape, d0, **kwargs):
	# lager et radialt/sirkulært filter med gitt funksjon
	# filteret lager en sirkel midt på bildet
	(h, w) = shape
	(y, x) = numpy.mgrid[:h, :w]
	# senterpixelen er ned mot høyre i bilder med partall som høyde/bredde
	distance = numpy.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2)
	return function(d=distance, d0=d0, **kwargs)

def linear_filter(function, shape, d0, vertical, **kwargs):
	# lager et rett filter med en gitt funksjon
	# filteret lager en vertikal eller horisontal stripe midt på bildet
	(h, w) = shape
	(hor_o, ver_o) = numpy.mgrid[:h, :w]
	hor = numpy.abs(hor_o - h // 2)
	ver = numpy.abs(ver_o - w // 2)
	out = function(d=ver if vertical else hor, d0=d0, **kwargs)
	# lar midten slippe gjennom - unngår at bildet mister flate områder
	out *= 1 - function(d=hor if vertical else ver, d0=d0, **kwargs)
	return out

def freq_filter(freq, f_filter, **kwargs):
	# filtrerer et bilde i frekvensplanet
	if "sharpen" in kwargs and kwargs["sharpen"] is True:
		f_filter = (1.0 + kwargs["k"] * f_filter)
	out = freq * f_filter
	return out

def fft(image):
	# konverterer et bilde til frekvensplanet med FFT
	(h, w) = image.shape
	pad_shape = (int(h * PF), int(w * PF))
	out = dft_trans(image)
	out = zero_pad(out, pad_shape)
	out = numpy.fft.fft2(out)
	return out

def ifft(image):
	# konverterer et bilde fra frekvensplanet med iFFT
	(h, w) = image.shape
	out = numpy.fft.ifft2(image)
	out = out[:int(h/PF), :int(w/PF)]
	#out = dft_trans(numpy.real(out))
	out = numpy.absolute(out) # ekvivalent
	out = numpy.clip(out, 0, 1)
	return out

img = util.img_as_float(io.imread("lena.png", as_gray=True))
(h, w) = img.shape
(h2, w2) = (int(h * PF), int(w * PF))

freq = fft(img)
#f_filter = linear_filter(butterworth, (h2, w2), 20, vertical=True, n=2)
f_filter = radial_filter(gauss, (h2, w2), 0.1*w2, n=2)
filtered = freq_filter(freq, f_filter, sharpen=False, k=1.0)
out = ifft(filtered)
#out = f_filter
#out = show_freq(filtered)

io.imsave("output.png", util.img_as_ubyte(out))