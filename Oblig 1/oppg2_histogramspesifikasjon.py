"""
Program som matcher et histogrammet i et bilde til et annet histogram.
Det andre histogrammet kan spesifiseres via et annet histogram-bilde,
 et 256*256 binært bilde, eller en tekstfil med 256 linjer.
Fungerer kun med 8-bits gråskalabilder.

Programmet kjøres slik:
python oppg1_histogramspesifikasjon.py input.png output.png hist.png "histogram"

Parametre:
1: Input-bilde
2: Output-bilde
3: Input-histogram
4: Metode
	"image": input-histogram er et bilde
	"histogram": input-histogram er et 256*256 binært bilde
	"text": input-histogram er en tekstfil med 256 linjer
"""
from skimage import io, util
import numpy
import sys

def hist_generate(image, levels):
	# genererer et histogram-array basert på et bilde.
	# hver index beskriver antall pixler med den intensiteten.
	out = numpy.zeros(levels)
	for pixel in numpy.nditer(image):
		out[pixel] += 1
	return out

def hist_equalize(hist):
	# normaliserer og akkummulerer et histgram
	hist_norm = hist / numpy.sum(hist)
	out = numpy.add.accumulate(hist_norm)
	return out

def gen_image(hist, height):
	# hjelpefunksjon som lager et 256x256 bilde av et histogram
	hist = numpy.round(hist * height / numpy.max(hist)).astype(numpy.int)
	out = numpy.ones((height, hist.size))
	for i, val in enumerate(hist):
		out[height-val:height, i] = 0.0
	return out

def parse_hist_image(image):
	# regner ut histogrammet gitt av et bilde.
	# Hver kolonne tilsvarer én intensitet. Den fungerer enkelt ved å telle
	#  opp piksler som er mellom 0 og 0.5 (svarte) i hver enkelt kolonne.
	(height, width) = image.shape
	out = numpy.zeros(width)
	for i in range(width):
		out[i] = sum(1 for j in image[:, i] if j <= 0.5)
	return out

def hist_match(hist_eq, target_eq):
	# regner ut matching mellom bildets histogram og histogrammet som skal matches.
	assert hist_eq.size == target_eq.size, "Histogrammene er ikke like store"
	out = numpy.zeros(hist_eq.size, dtype=numpy.ubyte)
	for i in range(hist_eq.size):
		# for hvert nivå i det akkumulerte histogrammet finner man det
		# nivået i det andre akkumulerte histogrammet med samme verdi
		# lagrer dette i en tabell
		out[i] = numpy.abs(target_eq - hist_eq[i]).argmin()
	return out

def main():
	levels = 256
	height = 256
	img = util.img_as_ubyte(io.imread(sys.argv[1], as_gray=True))
	img_save = sys.argv[2]
	method = sys.argv[4].lower()
	methods = ["image", "histogram", "text"]
	assert method in methods, "Parameter 4 må være enten 'image', 'histogram' eller 'text'"
	if method == methods[0]:
		hist_target_input = util.img_as_ubyte(io.imread(sys.argv[3], as_gray=True))
		hist_target = hist_generate(hist_target_input, levels)
	if method == methods[1]:
		hist_target_input = util.img_as_float(io.imread(sys.argv[3], as_gray=True))
		hist_target = parse_hist_image(hist_target_input)
	if method == methods[2]:
		hist_target_input = open(sys.argv[3], 'r')
		hist_target = numpy.uint(hist_target_input.readlines())
		assert len(hist_target) == levels, "Tektsfila må være {} linjer lang.".format(levels)

	hist = hist_generate(img, levels) # bilde-histogram
	hist_eq = hist_equalize(hist) # utjevnet histogram
	hist_target_eq = hist_equalize(hist_target) # utjevnet match-histogram
	hist_match_table = hist_match(hist_eq, hist_target_eq) # match-tabell
	img_spec = hist_match_table[img] # endelig bilde
	
	"""
	img_eq_map = numpy.ubyte(255 * hist_eq)
	img_eq = img_eq_map[img]
	#img_spec = spec_map[img_eq]
	hist_target_map = numpy.ubyte(255 * hist_target_eq)
	spec_map = numpy.zeros(256, dtype="ubyte")
	#spec_map = numpy.full(256, 255, dtype="ubyte")
	for i in range(len(spec_map)):
		for j in range(len(hist_target_map)):
			if hist_target_map[j] >= i + 1:
				spec_map[i] = j
				break
			"""
	
	#print(hist_target_map)
	#print(spec_map)
	#img_spec = gen_image(spec_map, height)
	#img_spec = spec_map#[img_eq]
	img_spec = gen_image(hist_target_eq, height)

	# bilder av histogrammene
	#hist_img = gen_image(hist, height)
	#hist_eq_img = gen_image(hist_eq, height)
	#hist_img_spec = gen_image(hist_generate(img_spec, levels), height)
	#hist_target_img = gen_image(hist_target, height)
	#hist_target_eq_img = gen_image(hist_target_eq, height)

	io.imsave(img_save, util.img_as_ubyte(img_spec))
	#io.imshow(img_spec)
	#io.show()
	
if __name__ == "__main__":
	main()