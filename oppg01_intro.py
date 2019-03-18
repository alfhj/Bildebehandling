import skimage
from skimage import io
#from skimage.viewer import ImageViewer

#io.use_plugin("imageio")
img = io.imread("bilder/einstein.tif")
newimg = img[50:600, 100:400]
print(newimg.shape)
#viewer = ImageViewer(newimg)
#viewer.show()
#io.imsave("test.tif", newimg)
io.imshow(newimg)
io.show()