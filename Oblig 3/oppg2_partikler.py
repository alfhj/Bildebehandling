from skimage import io, util, morphology
import numpy
import scipy

img = io.imread("bilder/2_particles.png", as_gray=True).astype(bool)
se = io.imread("bilder/2_particle.png", as_gray=True).astype(bool)
(h, w) = img.shape
(sh, sw) = se.shape

# finner punkter hvor SE finnes
hit = scipy.ndimage.morphology.binary_erosion(img, se)
miss = scipy.ndimage.morphology.binary_erosion(numpy.invert(img), numpy.invert(se))
single = hit & miss
overlapping = hit ^ single
proc = single

# fyller punktene med SE for å illustrere
out = proc.copy()
for i in range(h):
	for j in range(w):
		if proc[i, j] == True:
			out[i-sh//2:i+sh//2+1, j-sw//2:j+sw//2+1] |= se
#out ^= img
out = proc

out = (out * 255).astype(numpy.uint8)
io.imsave("bilder/output.png", out)