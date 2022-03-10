import cv2
import getopt
import math
import multiprocessing as mp
import numpy as np
import os
import sys
import Imath
import OpenEXR

deg2rad = lambda deg: deg / 180.0 * np.pi
verbose = False

def import_exr(file_path: str):
    # source: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)

    exr_file = OpenEXR.InputFile(file_path)
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    try:
        channel = exr_file.channel('R', PIXEL_TYPE)
    except TypeError:
        channel = exr_file.channel('Y', PIXEL_TYPE)
    depth = np.frombuffer(channel, dtype=np.float32)
    depth.shape = (height, width)
    return depth

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def depth_to_disparity(depth: np.ndarray, baseline: float, focal_length=-1, scale=1.0):
    invalid_mask_a = (depth <= 0)
    invalid_mask_b = (depth != depth)
    invalid_mask_c = (depth == np.inf)

    invalid_mask_ab = np.bitwise_or(invalid_mask_a, invalid_mask_b)
    invalid_mask = np.bitwise_or(invalid_mask_ab, invalid_mask_c)
    depth_cpy = np.copy(depth)
    depth_cpy *= scale
    depth_cpy[invalid_mask] = 1 # to avoid division by 0

    if focal_length <= 0:
        focal_length = np.min(depth.shape)/np.pi

    disparity = 1 / depth_cpy
    disparity = disparity * focal_length * baseline
    disparity[invalid_mask] = 0
    disparity[disparity < 0] = 0
    disparity[disparity >= 2**8] = 0

    disparity = disparity.astype(np.uint8)
    return disparity

def run(args):
    global verbose
    input_files = args[0]
    outdir = args[1]
    width = args[2]
    height = args[3]
    resize_width = 1024
    resize_height = 1024
    resize_dim = (resize_width, resize_height)
    pos = 0
    verbose = True

    for input_file in input_files:
        if verbose:
            print("processing image [" + str(pos + 1) + "/" + str(len(input_files)) + "]               \r", end="")
        img = import_exr(input_file)
        if img.shape[:2] != (height, width):
            print()
            print("Diverging input resolution!")
            help()
            exit(1)

        fov_omni = np.pi
        width = img.shape[0]
        f_omni = width / fov_omni

        basename = os.path.basename(input_file)
        root_ext = os.path.splitext(basename)
        output_file = outdir + "/" + root_ext[0][0:7] + '_disp' + '.webp'
        disparity  = depth_to_disparity(img,baseline = 0.3,focal_length =f_omni, scale=1.0)
        imgu8 = convert(disparity, 0, 255, np.uint8)
        #resized = cv2.resize(imgu8,resize_dim, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_file, imgu8)
        pos += 1


def help():
    print("usage: python3 to_pano.py [<option> [<option> [...]]]")
    print("<option>::")
    print("               --help           print this help message")
    print("    -i <arg> | --indir=<arg>    provide the input directory [mandatory]")
    print("    -o <arg> | --outdir=<arg>   provide the output directiory [mandatory]")
    print("")
    print("All input files must have the same resolution.")


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:w:h:f:av",
                                   ["help", "indir=", "outdir=", "width=", "height="])
    except getopt.GetoptError as err:
        help()
        exit(1)

    indir = None
    outdir = None

    for o, a in opts:
        if o == "--help":
            help()
            exit(0)
        elif o in ("-o", "--outdir"):
            outdir = a
        elif o in ("-i", "--indir"):
            indir = a
        else:
            print("unknown option:", o)
            help()
            exit(1)

    if indir is None:
        print("The input directory is missing!")
        help()
        exit(1)

    if outdir is None:
        print("The output directory is missing!")
        help()
        exit(1)

    input_files = []
    for root, dirs, files in os.walk(indir):
        for name in files:
            input_files.append(os.path.join(root, name))
        break

    if len(input_files) == 0:
        print("Input files missing!")
        help()
        exit(1)

    img_tmp = import_exr(input_files[0])
    height = img_tmp.shape[0]
    width = img_tmp.shape[1]

    num_files = len(input_files)
    num_threads = 16
    pool = mp.Pool(num_threads)
    num_files_per_thread = int(math.ceil(num_files / num_threads))
    all_args = []
    for i in range(num_threads):
        thread_files = input_files[i * num_files_per_thread: (i + 1) * num_files_per_thread]
        arguments = [thread_files, outdir,width,height]
        all_args.append(arguments)
    pool.map(run, all_args)
    pool.close()
    #
    # if verbose:
    #     print()
    #     print("Finished")


if __name__ == "__main__":
    main()
