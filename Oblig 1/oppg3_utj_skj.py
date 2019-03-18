"""
Utjevning og skjerping av bilder

Parametrene etter bildet skrives som en kommaseparert liste.
Den første verdien bestemmer metoden som skal benyttes.
Hvis man legger på flere sett med parametre, vil programmet
 utføre operasjonene etter hverandre.

Programmet kjøres slik:
python oppg3_utj_skj.py "lena.tif" "b1,size=5" "s2,str=0.25"
Denne vil utgjevne bildet med en 5x5 1-matrise,
 og deretter skjerpe den med laplace med styrke 0.25.

Parametre:
1: Input-bilde (tekst)
2: Output-bilde (tekst)
3++: Liste over parametre:
    b[tall] eller s[tall] for sharpen eller blur
        tallet bestemmer hvilken metode som skal brukes:
        b1: Gjennomsnitt
        b2: Vektet gjennomsnitt
        b3: Median
        b4: Gaussian
        b5: Gaussian med laplace-maske
        b6: Bilateral
        s1: Sobel-skjerping
        s2: Laplace-skjerping 1
        s3: Laplace-skjerping 2
        s4: Unsharp
    size (3): Størrelse på filter (heltall, oddetall)
    str (strength, 1.0): Styrke på skjerping (desimaltall)
    sigs (sigma_s, 0.67): Sigma som beskriver styrke til gaussian-filteret
     høyere verdier gir mer utjevning
    sigr (sigma_r): Sigma som beskriver styrke til bilateral-filteret
     lavere verdier gir skarpere bilder
"""
from skimage import color, io, util
import oppg1_konvolusjon as oppg1
import numpy
import scipy
import sys
import profile

# filtre:
sobelx = numpy.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobely = numpy.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
laplace1 = numpy.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])
laplace2 = numpy.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

def gauss_function(dist2, sigma):
    # konstantleddet er unødvendig siden filteret blir normalisert til slutt.
    #const = 1 / (2 * numpy.pi * sigma ** 2)
    #out = const * numpy.exp(-dist2 / (2 * sigma ** 2))
    out = numpy.exp(-dist2 / (2 * sigma ** 2))
    return out

def gauss_filter(size, sigma_s):
    # regner ut n*n-matrise med gaussian vekter 
    out = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            s = i - size // 2
            t = j - size // 2
            dist = s ** 2 + t ** 2
            out[i, j] = gauss_function(dist, sigma_s)
    return out

def average_filter(size):
    # n*n-matrise med enere
    out = numpy.ones((size, size))
    return out

def weighted_filter(size):
    # n*n-matrise med 2-er-potenser
    w = size // 2
    out = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            out[i, j] = 1 << (w - abs(i - w)) * 1 << (w - abs(j - w))
    return out
    
def apply_filter(image, c_filter):
    # utfører konvolusjon på et bilde
    filter_sum = numpy.sum(c_filter)
    c_filter = c_filter / (1 if filter_sum == 0 else filter_sum) # normaliserer
    out = scipy.ndimage.convolve(image, c_filter)
    return out

def sobel(image):
    # genererer sobel-maske (gradienten) til et bilde
    sx = apply_filter(image, sobelx)
    sy = apply_filter(image, sobely)
    out = numpy.sqrt(sx ** 2 + sy ** 2)
    return out
    
def sharpen(image, mask, strength):
    # legger en maske over et bilde for å skjerpe
    out = image + strength * mask
    return out

def median(image, size):
    # utfører median-filtrering av et bilde med n*n-vindu
    (h, w) = image.shape
    out = numpy.zeros(image.shape)
    for i in range(h):
        for j in range(w):
            i1 = max(0, i - size // 2)
            i2 = min(h, i + size // 2 + 1)
            j1 = max(0, j - size // 2)
            j2 = min(w, j + size // 2 + 1)
            subimg = image[i1:i2, j1:j2]
            out[i, j] = numpy.median(subimg)
    return out

def bilateral_filter(segment, g_filter, sigma_r):
    # regner ut bilateral-filter for et bilde-segment
    # endrer et gaussian-filter ved å legge på intensitet-vekter
    size = segment.shape[0]
    out = g_filter.copy()
    center_value = segment[size//2, size//2]
    for i in range(size):
        for j in range(size):
            val_range = (segment[i, j] - center_value) ** 2
            out[i, j] *= gauss_function(val_range, sigma_r)
    return out

def bilateral(image, size, sigma_s, sigma_r):
    # utfærer bilateral-filtrering på et bilde
    out = image.copy()
    (h, w) = image.shape
    image = oppg1.pad(image, size//2, "mirror")
    g_filter = gauss_filter(size, sigma_s)
    for i in range(h):
        for j in range(w):    
            segment = image[i:i+size, j:j+size]
            c_filter = bilateral_filter(segment, g_filter, sigma_r)
            c_filter = oppg1.normalize_filter(c_filter)
            out[i, j] = numpy.sum(numpy.multiply(segment, c_filter))
    return out

def args_parse(args_input, image):
    # parser parametrene og genererer default verdier
    args_split = args_input.split(',')
    args = dict(item.split('=') for item in args_split[1:])
    args["code"] = args_split[0]
  
    if "size" in args:
        args["size"] = numpy.int(args["size"])
        assert args["size"] >= 1 and args["size"] % 2 == 1, \
            "Størrelsen må være positiv og oddetall."
    else: args["size"] = 3

    if "str" in args: args["str"] = numpy.float(args["str"])
    else: args["str"] = 1.0

    if "sigs" in args: args["sigs"] = numpy.float(args["sigs"])
    else:
        # default sigma_s er avhengig av filterstørrelsen
        args["sigs"] = (args["size"] + 1) / 6 
        #(h, w) = image.shape # kan være avhengig av bildestørrelsen
        #args["sigs"] = 0.001 * numpy.sqrt(h ** 2 + w ** 2)

    if "sigr" in args: args["sigr"] = numpy.float(args["sigr"])
    else:
        # default sigma_r er avhengig av gradienten til bildet
        if len(image.shape) == 3: image = color.rgb2gray(image)
        args["sigr"] = 0.4 * numpy.median(sobel(image))
    
    return args

def action(image, args):
    # utfører filtrering på et bilde ut i fra parametre
    method = args["code"][0]
    number = int(args["code"][1:])

    assert method == "b" or method == "s", "Metode må starte på 's' eller 'b'"
    if method == "b":
        assert number >= 1 and number <= 6, "Tallet i metode må være mellom 1 og 6"
        if number == 1:
            print("Utjevner bildet med gjennomsnitt.")
            c_filter = average_filter(args["size"])
            out = apply_filter(image, c_filter)
        if number == 2:
            print("Utjevner bildet med vektet gjennomsnitt.")
            c_filter = weighted_filter(args["size"])
            out = apply_filter(image, c_filter)
        if number == 3:
            print("Utjevner bildet med median-filtrering.")
            out = median(image, args["size"])
        if number == 4:
            print("Utjevner bildet med gauss-filter.")
            c_filter = gauss_filter(args["size"], args["sigs"])
            out = apply_filter(image, c_filter)
        if number == 5:
            print("Utjevner bildet med gauss-filter og legger på laplace-maske.")
            c_filter = gauss_filter(args["size"], args["sigs"])
            blur = apply_filter(image, c_filter)
            mask = apply_filter(image, laplace1)
            out = sharpen(blur, mask * image, args["str"]) # legger til skarpe kanter.
        if number == 6:
            print("Utjevner bildet med bilateral filtrering (dette vil ta tid).")
            out = bilateral(image, args["size"], args["sigs"], args["sigr"])

    if method == "s":
        assert number >= 1 and number <= 4, "Tallet i metode må være mellom 1 og 4"
        if number == 1:
            print("Skjerper bildet med sobel-maske.")
            mask = sobel(image)
        if number == 2:
            print("Skjerper bildet med laplace-maske.")
            mask = apply_filter(image, laplace1)
        if number == 3:
            print("Skjerper bildet med laplace-maske som tar med diagonale pixler.")
            mask = apply_filter(image, laplace2)
        if number == 4:
            print("Skjerper bildet med gauss-maske (highboost/unsharp).")
            c_filter = gauss_filter(args["size"], args["sigs"])
            blur = apply_filter(image, c_filter)
            mask = image - blur
        out = sharpen(image, mask, args["str"])
    
    out = numpy.clip(out, 0, 1)
    return out

def main():
    img = util.img_as_float(io.imread(sys.argv[1]))
    img_save = sys.argv[2]
    sz = img.shape
    out = img.copy()

    for arg in sys.argv[3:]:
        args = args_parse(arg, img)
        print("Parametre: {}".format(args))
        if len(sz) == 3: # hvis bildet er i RGB-format
            for i in range(sz[2]): # filtrer hver kanal
                out[:, :, i] = action(out[:, :, i], args)
        else:
            out = action(out, args)

    io.imsave(img_save, util.img_as_ubyte(out))
    #io.imshow(out)
    #io.show()

if __name__ == "__main__":
    #profile.run("main()")
    main()