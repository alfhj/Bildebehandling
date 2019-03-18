import skimage
import numpy
from skimage import io
from skimage import util

K = 200
img = io.imread("itd33517_examples/M101.jpg")
sum_image = numpy.ndarray(img.shape)
images = []

for i in range(K):
    sum_image += util.random_noise(img, mode="gaussian", var=0.1)
    if i in [4, 9, 19, 49, 199]:
        images.append(sum_image / (i + 1))

images.append(util.img_as_float(img))
#print(numpy.concatenate(images, axis=1).shape)
#print(img.shape)
#new_image = numpy.stack(numpy.asmatrix(new_images), axis=0)
#io.imshow(numpy.asanyarray(images))
io.imshow(numpy.concatenate(images, axis=1))
io.show()