from skimage import io, util
import numpy
import scipy

img_names = ["o1_1.png", "o1_2.png", "o2_1.png", "o3_1.png", "o3_2.png", "o3_3.png", "o4_1.png", "o4_2.png"]
imgs = [util.img_as_float(io.imread("itd33517_examples/{}".format(img), as_gray=True)) for img in img_names]

se = numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
data = []
for (name, img) in zip(img_names, imgs):
	ero = scipy.ndimage.morphology.binary_erosion(img, se)
	edge = img - ero
	area = numpy.sum(img)
	circ = numpy.sum(edge)
	
	if area < 25 and circ < 25:	obj = "1"
	elif area > 25 and circ < 25: obj =  "3/4"
	elif area > 25 and circ > 25: obj = "2"
	else: obj = "unknown"
	data.append([name, area, circ, obj])
	#print("{}: area={}, circumference={}, object={}".format(name, area, circ, obj))

print('\n'.join("{}: area={}, circumference={}, object={}".format(*line) for line in data))
#print(data)
#out = (out * 255).astype(numpy.uint8)
#io.imsave("output.png", out)