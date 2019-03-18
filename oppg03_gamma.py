from skimage import io, util
import sys
import numpy

img = util.img_as_float(io.imread(sys.argv[1], as_gray=True))
c = numpy.float(sys.argv[2])
gamma = numpy.float(sys.argv[3])
height, width = img.shape
out = numpy.zeros(img.shape, dtype=numpy.float)
for i in range(height):
    for j in range(width):
        out[i,j] = c * img[i,j] ** gamma

io.imshow(numpy.clip(out, 0, 1))
io.show()