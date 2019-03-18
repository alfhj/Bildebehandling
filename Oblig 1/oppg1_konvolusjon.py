"""
Program som utfører konvolusjon på et bilde. Bildet kan ha flere kanaler.

Programmet kjøres slik:
python oppg1_konvolusjon.py input.png output.png "copy" 1

Parametre:
1: Input-bilde (tekst)
2: Output-bilde (tekst)
3: Padding-metode (tekst)
    "zero": kantene fylles med 0-ere
    "copy": kantene fylles med samme verdi som nærmeste pixel i bildet
    "mirror": kantene speiles om bildekanten
4: Filter (heltall), filteret/masken som skal benyttes
"""
from skimage import io, util
import numpy
import sys

def filters(n):
    # henter ut et gitt filter
    coll = [numpy.array([ # 1: Identitet
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]), numpy.array([ # 2: Box blur
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]), numpy.array([ # 3: Vektet blur
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]), numpy.array([ # 4: Laplace-skjerping
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]), numpy.array([ # 5: Laplace med diagonal
        [1, 1, 1],
        [1, -9, 1],
        [1, 1, 1]
    ]), numpy.array([ # 6: Sobel kant i horisontal
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]), numpy.array([ # 7: "Gaussian" skjerping, altså unsharp
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, -476, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])]
    assert (n > 0 and n <= len(coll)), "Tallet må være mellom 1 og {}".format(len(coll))
    out = coll[n-1]
    return out

def pad(image, width, method):
    # padder et bilde med en av tre metoder:
    # Metode 1 (zero): utvid med 0-verdier
    # Metode 2 (copy): utvid med samme verdi som ytterste (nærmeste) pixel
    # Metode 3 (mirror): utvid med refleksjonen av kantpixler

    method = method.lower()
    methods = ["zero", "copy", "mirror"]
    assert method in methods, "Error: 'method' må være 'zero', 'copy' eller 'mirror'"

    (h, w) = image.shape

    #lager øvre og nedre stripe
    if method == methods[0]:
        top_strip = numpy.zeros((width, w))
        bottom_strip = top_strip.copy()
    if method == methods[1]:
        top_strip = numpy.repeat(image[:1, :], width, axis=0)
        bottom_strip = numpy.repeat(image[-1:, :], width, axis=0)
    if method == methods[2]:
        top_strip = numpy.flip(image[0:width, :], axis=0)
        bottom_strip = numpy.flip(image[-width:, :], axis=0)

    out = numpy.concatenate((top_strip, image, bottom_strip), axis=0)

    # lager høyre og venstre stripe
    if method == methods[0]:
        left_strip = numpy.zeros((h + 2 * width, width))
        right_strip = left_strip.copy()
    if method == methods[1]:
        left_strip = numpy.repeat(out[:, :1], width, axis=1)
        right_strip = numpy.repeat(out[:, -1:], width, axis=1)
    if method == methods[2]:
        left_strip = numpy.flip(out[:, 0:width], axis=1)
        right_strip = numpy.flip(out[:, -width:], axis=1)

    # setter det hele sammen
    out = numpy.concatenate((left_strip, out, right_strip), axis=1)
    return out

def normalize_filter(c_filter):
    # normaliserer en filter (maske) slik at verdiene summerer til 1
    sum_filter = numpy.sum(c_filter)
    out = c_filter / (1 if sum_filter == 0 else sum_filter)
    return out
    
def apply_filter(segment, c_filter):
    # utfører konvolusjon på ett segment
    out = 0
    (h, w) = segment.shape
    for i in range(h):
        for j in range(w):
            out += segment[i, j] * c_filter[i, j]
    return out

def convolve(image, c_filter, pad_method):
    # utfører konvolusjon på et helt bilde
    fs = c_filter.shape
    assert fs[0] == fs[1] and fs[0] % 2 == 1, \
        "Filteret må være kvadratisk og oddetall stort"
    out = image.copy()
    c_filter = normalize_filter(c_filter)
    fw = fs[0] # størrelsen til filteret
    (h, w) = image.shape # størrelsen til det originale bildet
    image = pad(image, fw//2, pad_method)
    # går gjennom det paddede bildet
    # setter inn nye verdier i et nytt bilde
    for i in range(h):
        for j in range(w):
            segment = image[i:i+fw, j:j+fw]
            out[i, j] = apply_filter(segment, c_filter)
    out = numpy.clip(out, 0, 1) # sørger for at verdiene er innafor rekkevidde
    return out

def main():
    img = util.img_as_float(io.imread(sys.argv[1]))
    img_save = sys.argv[2]
    pad_method = sys.argv[3]
    filter_choice = filters(numpy.int(sys.argv[4]))
    sz = img.shape
    out = img.copy()
    if len(sz) == 3: # hvis bildet er i RGB-format
        for i in range(sz[2]): # utfør konvolusjon på hver kanal
            out[:, :, i] = convolve(img[:, :, i], filter_choice, pad_method)
    else:
        out = convolve(img, filter_choice, pad_method)
    io.imsave(img_save, util.img_as_ubyte(out))
    #io.imshow(numpy.concatenate((img, out), axis=1))
    #io.show()

if __name__ == "__main__":
    main()