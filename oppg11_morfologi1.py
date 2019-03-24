from skimage import io, util, morphology
import numpy
import scipy

#img = util.img_as_bool(io.imread("itd33517_examples/binary2.pbm", as_gray=True))
#img = util.img_as_bool(io.imread("bilder/letters-and-objects.tif", as_gray=True))
#elems = util.img_as_bool(io.imread("itd33517_examples/se_cross.pbm", as_gray=True))
elems = numpy.ones((20, 20), dtype=bool)
#ero = scipy.ndimage.morphology.binary_erosion(img, elems)
#open = morphology.binary_dilation(img, elems)
#bound = img * (1 - ero)
#bound = img ^ ero

comp = util.img_as_bool(io.imread("bilder/balls-with-reflections.tif"))
img = numpy.invert(comp)
seeds = util.img_as_bool(io.imread("washers_seeds.png", as_gray=True))
seeds_old = None

count = 0
while not numpy.array_equal(seeds, seeds_old):
	count += 1
	seeds_old = seeds.copy()
	seeds = scipy.ndimage.morphology.binary_dilation(seeds, elems)
	seeds *= comp

print(count)
out = img + seeds
io.imsave("output.png", out*255)
#io.imshow(out)
#io.show()