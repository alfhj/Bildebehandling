from skimage import feature, filters, io, util, dtype_limits
import numpy
import scipy

img = util.img_as_float(io.imread("bilder/barbara_crop.png", as_gray=True))
(h, w) = img.shape

# finner utjevnet gradientbilde
gauss = filters.gaussian(img, 0.01)
sv = filters.sobel_v(img)
sh = filters.sobel_h(img)
mag = filters.sobel(gauss)
ang = sv / sh
ang_abs = numpy.abs(ang)

# nonmax suppression
padded = numpy.pad(mag, 1, mode="constant", constant_values=0)
non_max = numpy.zeros((h, w))
for i in range(h):
	for j in range(w):
		sub = padded[i:i+3, j:j+3]
		if ang_abs[i, j] >= numpy.sqrt(2) + 1: neigh = (sub[1, 0], sub[1, 2])
		elif ang[i, j] >= numpy.sqrt(2) - 1: neigh = (sub[0, 0], sub[2, 2])
		elif ang[i, j] <= 1 - numpy.sqrt(2): neigh = (sub[0, 2], sub[2, 0])
		else: neigh = (sub[0, 1], sub[2, 1])
		if mag[i, j] >= max(neigh): non_max[i, j] = mag[i, j]

# hysterisis thresholding
th = 0.03 # øvre grense
tl = 0.015 # nedre grense
nm_high = non_max >= th # sterke kanter
nm_low = (non_max >= tl) & (non_max < th) # svake kanter
nm_low_pad = numpy.pad(nm_low, 1, mode="constant", constant_values=False)
to_check = numpy.transpose(nm_high.nonzero()) # koordinater til sterke kanter som gjenstår
final = numpy.zeros((h, w), dtype=numpy.bool) # endelig bilde
# går gjennom de sterke kantene og legger til de svake kantene
#  som er i 8-naboskap helt til alle kantene er besøkt
while numpy.size(to_check, 0) != 0:
	i = to_check[0][0]
	j = to_check[0][1]
	weak = numpy.transpose(nm_low_pad[i:i+3, j:j+3].nonzero()) + (i, j) - 1 # finner koordinater til svake kanter rundt
	to_check = numpy.append(to_check[1:], weak, 0) # sletter nåverende kant og legger til nye kanter
	nm_low_pad[i:i+3, j:j+3] = False # markerer svake kanter som besøkt
	final[i, j] = True # oppdaterer output

canny_ref = feature.canny(img, 2.0)
#out = nm_low * 0.5 + nm_high
out = final

out = (out * 255).astype(numpy.uint8)
io.imsave("output.png", out)