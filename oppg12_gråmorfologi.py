from skimage import io, util, morphology
import numpy

# d: diameter
def circular_se(d):
	(y, x) = numpy.mgrid[:d, :d]
	dist = numpy.sqrt((y - d / 2 + 0.5) ** 2 + (x - d / 2 + 0.5) ** 2)
	circ = (dist <= d / 2 - 0.25).astype(bool)
	return circ

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

def open(image, se):
	eroded = morph(image, se, True)
	out = morph(eroded, se, False)
	return out

def close(image, se):
	dilated = morph(image, se, False)
	out = morph(dilated, se, True)
	return out

def blur(image, se):
	out = open(image, se)
	out = close(out, se)
	return out

def blur1(image, se):
	out = close(image, se)
	out = open(out, se)
	return out

def gradient(image, se):
	eroded = morph(image, se, True)
	dilated = morph(image, se, False)
	out = dilated - eroded
	return out

def tophat(image, se):
	opened = open(image, se)
	out = image - opened
	return out

def bottomhat(image, se):
	closed = close(image, se)
	out = closed - image
	return out

img = util.img_as_float(io.imread("bilder/hurricane-katrina-eye.tif"))

se = circular_se(11)
print(se)
out = bottomhat(img, se)
print(numpy.min(out), numpy.max(out))

io.imsave("output.png", out)