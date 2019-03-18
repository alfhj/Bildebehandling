from skimage import io, util, color
import sys
import numpy

def hist_generate(image, bins):
    out = [0] * bins
    for pixel in numpy.nditer(image):
        out[pixel] += 1
    return out

def hist_equalize(hist):
    hist_norm = hist / numpy.sum(hist)
    out = numpy.add.accumulate(hist_norm)
    return out

img = util.img_as_ubyte(io.imread(sys.argv[1], as_gray=True))
levels = 256
hist = hist_generate(img, levels)
hist_eq = hist_equalize(hist)
img_new = numpy.ubyte((levels - 1) * hist_eq[img])
io.imshow(img_new)
io.show()