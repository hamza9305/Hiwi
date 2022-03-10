import sys
import array
import OpenEXR
import Imath
import numpy as np
from PIL import Image
import os


# input_dir = 'F:\Hiwi\THEOStereo\Read_exr_files/sample (1).exr'
# output_dir = 'F:\Hiwi\THEOStereo\Read_exr_files\cylindrical\sample.jpg'


# Open the input file
def convert(exrfile,jpgfile):
    file = OpenEXR.InputFile(exrfile)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    rgbf = [Image.frombytes("F", size, file.channel(c, pt)) for c in "RGB"]

    extrema = [im.getextrema() for im in rgbf]
    darkest = min([lo for (lo,hi) in extrema])
    lighest = max([hi for (lo,hi) in extrema])
    scale = 255 / (lighest - darkest)
    def normalize_0_255(v):
        return (v * scale) + darkest
    rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
    Image.merge("RGB", rgb8).save(jpgfile)



def main():
    indir = '/mnt/data/haahm/THEOStereo/theostereo_Omni/train/depth_exr_abs'
    outdir = '/mnt/data/haahm/THEOStereo/theostereo_Omni/depth_exr_abs'

    # image = Image.open(open_img)
    input_files = []
    for root, dirs, files in os.walk(indir):
        for name in files:
            input_files.append(os.path.join(root, name))

    for files in input_files:
        basename = os.path.basename(files)
        root_ext = os.path.splitext(basename)
        output_file = outdir + "/" + root_ext[0] + '.jpg'
        convert(files,output_file)



if __name__ == "__main__":
   main()
