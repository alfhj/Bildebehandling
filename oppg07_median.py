from skimage import io, util
import sys
import scipy
import numpy
from math import pi, exp

def center(x, y, size):
    return None

def apply_function(image, x, y, size, function):
    (height, width) = image.shape
    values = []
    coords = range(-size//2, size//2+1)
    for i in coords:
        for j in coords:
            x1 = x + i
            y1 = y + j
            if x1 < 0 or y1 < 0 or x1 >= width or y1 >= height:
                continue
            values.append(image[y1, x1])
    out = numpy.median(values)
    return out
               

img = util.img_as_float(io.imread(sys.argv[1], as_gray=True))

size = numpy.int(sys.argv[2])
if size < 1 or size % 2 == 0:
    print("Størrelsen er på feil format")
    sys.exit(1)

#med = scipy.ndimage.filters.generic_filter(img, numpy.median, size)
vals = []
count = 0
(height, width) = img.shape
pixels = height * width
out = numpy.zeros(img.shape)
for y in range(height):
    for x in range(width):
        #out[y, x] = apply_function(img, x, y, size, None)
        count += 1
        x1 = max(0, x - size // 2)
        x2 = min(width, x + size // 2 + 1)
        y1 = max(0, y - size // 2)
        y2 = min(height, y + size // 2 + 1)
        subimg = img[y1:y2, x1:x2]
        out[y, x] = numpy.median(subimg)
        if count % 2500 == 0: print(str(round(100*(y*width+x)/pixels,2)) + "%")
        #if count == 50000: break
    else: continue
    break

#print(vals)
io.imshow(out)
io.show()