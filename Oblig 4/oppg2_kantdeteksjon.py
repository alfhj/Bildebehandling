from skimage import feature, io, util
import numpy
import scipy

# utfører konvolusjon på et bilde med et filter
#  normaliserer først vektene slik at de summerer til 1
#  med mindre de summerer til 0 allerede
def convolve(image, c_filter):
	f_sum = numpy.sum(c_filter)
	c_filter = c_filter / (1 if abs(f_sum) < 1e-10 else f_sum)
	out = scipy.ndimage.filters.convolve(image, c_filter)
	return out

# genererer et gaussian filter
def gauss_filter(w, sigma):
	(y, x) = numpy.mgrid[:w, :w]
	dist2 = (y - w // 2) ** 2 + (x - w // 2) ** 2 # avstand fra sentrum
	out = numpy.exp(-dist2 / (2 * sigma ** 2)) # normalfordeling uten konstantledd - skal normaliseres til slutt
	return out
	
# genererer et Laplacian of Gaussian (LoG) filter
#  utfører utjevning og andrederivasjon i samme steg
def laplace_gauss_filter(w, sigma):
	(y, x) = numpy.mgrid[:w, :w]
	dist2 = (y - w // 2) ** 2 + (x - w // 2) ** 2 # avstand fra sentrum
	del2 = (dist2 - 2 * sigma ** 2) / (sigma ** 4) # laplacian
	out = del2 * numpy.exp(-dist2 / (2 * sigma ** 2)) # LoG
	out -= numpy.sum(out) / w ** 2 # normaliserer slik at summen blir 0
	return out

# utfører zero-crossing til bruk i Marr-Hildreth kantdeteksjon
def zero_crossing(image):
	padded = numpy.pad(image, 1, mode="constant", constant_values=0)
	out = numpy.zeros((h, w), dtype=float)
	for i in range(h):
		for j in range(w):
			diag = [(padded[i+0, j+0], padded[i+2, j+2]),
				(padded[i+0, j+1], padded[i+2, j+1]),
				(padded[i+0, j+2], padded[i+2, j+0]),
				(padded[i+1, j+0], padded[i+1, j+2])]
			vals = [abs(p[0] - p[1]) for p in diag if p[0] * p[1] < 0]
			# henter største differanse mellom diagonale piklser med forskjellig fortegn
			if vals: out[i, j] = max(vals)
	return out

# genererer et sirkulært strukturerende element
def circular_se(d):
	(y, x) = numpy.mgrid[:d, :d]
	dist = numpy.sqrt((y - d / 2 + 0.5) ** 2 + (x - d / 2 + 0.5) ** 2)
	out = (dist <= d / 2 - 0.25).astype(bool)
	return out

# utfører erosjon eller utvidning på et gråskalabilde
def morph(image, se, erode=True):
	(ih, iw) = image.shape
	(seh, sew) = se.shape
	padded = util.pad(image, (seh//2, sew//2), mode="constant", constant_values=1 if erode else 0)
	out = numpy.zeros((ih, iw), dtype="float")
	for i in range(ih):
		for j in range(iw):
			subimg = padded[i:i+seh, j:j+sew]
			subset = subimg[se]
			out[i, j] = numpy.min(subset) if erode else numpy.max(subset)
	return out

img = util.img_as_float(io.imread("bilder/barbara_crop.png", as_gray=True))
(h, w) = img.shape
size = 15
sigma = 0.01
thr = 2000
gauss = gauss_filter(size, sigma)
blur = convolve(img, gauss)

# Sobel
sobelx = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobely = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sx = convolve(blur, sobelx)
sy = convolve(blur, sobely)
sm = numpy.sqrt((sx ** 2 + sy ** 2) / 2)
sobeledge = sm > thr

# Marr-Hildreth
lgauss = laplace_gauss_filter(size, sigma)
log = convolve(img, lgauss)
edges = zero_crossing(log)
marrhild = edges > thr

# Canny
canny = feature.canny(img, sigma)

# Morfologisk gradient
se = circular_se(3)
eroded = morph(blur, se, True)
dilated = morph(blur, se, False)
grad = dilated - eroded
morphgrad = grad > thr

out = marrhild
out = (out * 255).astype(numpy.uint8)
io.imsave("output.png", out)