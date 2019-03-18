from skimage import io, util
import sys
import numpy

def morph(image, elements, erode=True):
	(ih, iw) = image.shape
	(eh, ew) = elements.shape
	padded = util.pad(image, (eh//2, ew//2), mode="edge")
	out = numpy.zeros((ih, iw), dtype="bool")
	for i in range(ih):
		for j in range(iw):
			sub = padded[i:i+eh, j:j+ew]
			if erode:
				out[i, j] = numpy.array_equal(sub*elements, elements)
			else:
				out[i, j] = numpy.sum(sub*elements) > 0
	return out

#img = util.img_as_float(io.imread("itd33517_examples/binary2.pbm", as_gray=True))
img = util.img_as_float(io.imread("bilder/letters-and-objects.tif", as_gray=True))
elems = util.img_as_bool(io.imread("itd33517_examples/se_cross.pbm", as_gray=True))
#elems = numpy.ones((10, 10))
ero = morph(img, elems, True)
#open = morph(ero, elems, False)
bound = img - ero
out = open

#io.imsave("binary2_erosion.png", out)
io.imshow(out)
io.show()