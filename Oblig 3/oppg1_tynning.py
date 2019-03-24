from skimage import io, util
import numpy
import scipy

# Generer SE
C1 = numpy.array([[False, False, False], [False, True, False], [True, True, True]])
D1 = numpy.array([[True, True, True], [False, False, False], [False, False, False]])
C2 = numpy.array([[False, False, False], [True, True, False], [True, True, False]])
D2 = numpy.array([[False, True, True], [False, False, True], [False, False, False]])
B1 = []
B2 = []
# roterer SE og lagrer i liste
for k in range(4):
	B1.append(numpy.rot90(C1, k, (1, 0)))
	B2.append(numpy.rot90(D1, k, (1, 0)))
	B1.append(numpy.rot90(C2, k, (1, 0)))
	B2.append(numpy.rot90(D2, k, (1, 0)))

# morfologisk tynning
# bruker et SE for forgrunn og et for BG og kombinerer resultatet
def thin(image):
	out = image.copy()
	for (b1, b2) in zip(B1, B2):
		fg = scipy.ndimage.morphology.binary_erosion(out, b1, border_value=False)
		bg = scipy.ndimage.morphology.binary_erosion(numpy.invert(out), b2, border_value=True)
		out ^= fg & bg # kombinerer match i fg og bg og tar differansen med input
	return out

img = io.imread("bilder/1_morph.png", as_gray=True).astype(bool)
img = numpy.invert(img)

# utfører tynning helt til bildet ikke endrer seg
out = img
prev = None
count = 0
while not numpy.array_equal(out, prev):
	print(out.astype(int))
	count += 1
	prev = out.copy()
	out = thin(out)
print("{} iterasjoner".format(count))

out = ((1 - out) * 255).astype(numpy.uint8)
io.imsave("bilder/output.png", out)