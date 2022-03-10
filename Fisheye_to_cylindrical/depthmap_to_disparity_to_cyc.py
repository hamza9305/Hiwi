import cv2
import getopt
import math
import multiprocessing as mp
import numpy as np
import os
import sys
import Imath
import OpenEXR

verbose = False
deg2rad = lambda deg: deg / 180.0 * np.pi


def pano2wrld(width, height):
    f_cyl = 1
    y_max = f_cyl * np.pi / 2
    y_min = -y_max
    x_min = y_min
    x_max = y_max

    x_space = np.linspace(x_min, x_max, width)
    y_space = np.linspace(y_min, y_max, height)
    xm, ym = np.meshgrid(x_space, y_space)

    x_wrld = xm
    y_wrld = np.sin(ym / f_cyl)
    z_wrld = np.cos(ym / f_cyl)

    n = np.sqrt(np.square(x_wrld) + np.square(y_wrld) + np.square(z_wrld))
    x_wrld = np.expand_dims(x_wrld, axis=0)
    y_wrld = np.expand_dims(y_wrld, axis=0)
    z_wrld = np.expand_dims(z_wrld, axis=0)
    wrld = np.concatenate((x_wrld, y_wrld, z_wrld), axis=0)
    wrld /= n
    return wrld


def pano2omni(pano_width, pano_height, omni_width, omni_height):
    global verbose
    wrld = pano2wrld(pano_width, pano_height)
    v_dir = np.array([[0], [0], [1]])  # camera looks into z direction
    cos_thetas = np.transpose(np.transpose(wrld).dot(v_dir))
    # perfect fish eye -> $\rho(\theta) = \theta$
    rhos = np.arccos(cos_thetas)
    # normalize rho to [0, 1] instead of [0, pi]
    rhos /= (np.pi / 2)
    u = wrld[:2]
    u /= np.linalg.norm(u, axis=0)
    omni_coords_norm = u * rhos
    x_norm = omni_coords_norm[0, :]
    y_norm = omni_coords_norm[1, :]
    x_img = pano_width / 2 * x_norm + omni_width / 2
    y_img = pano_height / 2 * y_norm + omni_height / 2
    return x_img, y_img


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
    input_files = args[0]
    outdir = args[1]
    width = args[2]
    height = args[3]
    x_img = args[4]
    y_img = args[5]
    provide_aux_lines = args[6]
    resize_width = 1024
    resize_height = 1024
    resize_dim = (resize_width, resize_height)
    global verbose
    pos = 0
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

        img_cyl = cv2.remap(img, x_img, y_img, cv2.INTER_LINEAR)
        if provide_aux_lines:
            img_cyl[::20,:,:] = 0

        basename = os.path.basename(input_file)
        #print(basename)
        #exit(0)
        root_ext = os.path.splitext(basename)
        #print(root_ext[0][0:7])
        #exit(0)
        output_file = outdir + "/" + root_ext[0][0:7] + '_disp' + '.webp'
        #print(output_file)
        #exit(0)
        disparity  = depth_to_disparity(img_cyl,baseline = 0.3,focal_length =f_omni, scale=1.0)
        imgu8 = convert(disparity, 0, 255, np.uint8)
        resized = cv2.resize(imgu8,resize_dim, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_file, resized)
        pos += 1


def help():
    print("usage: python3 to_pano.py [<option> [<option> [...]]]")
    print("<option>::")
    print("               --help           print this help message")
    print("          -v | --verbose        activate verbose mode")
    print("    -i <arg> | --indir=<arg>    provide the input directory [mandatory]")
    print("    -w <arg> | --width=<arg>    provide the target width")
    print("    -h <arg> | --height=<arg>   provide the target height")
    print("    -o <arg> | --outdir=<arg>   provide the output directiory [mandatory]")
    print("")
    print("All input files must have the same resolution.")


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:w:h:f:av",
                                   ["help", "indir=", "outdir=", "width=", "height=", "aux", "verbose"])
    except getopt.GetoptError as err:
        help()
        exit(1)

    global verbose
    indir = None
    outdir = None
    pano_width = None
    pano_height = None
    provide_aux_lines = False

    for o, a in opts:
        if o == "--help":
            help()
            exit(0)
        elif o in ("-o", "--outdir"):
            outdir = a
        elif o in ("-i", "--indir"):
            indir = a
        elif o in ("-w", "--width"):
            pano_width = int(a)
        elif o in ("-h", "--height"):
            pano_height = int(a)
        elif o in ("-a", "--aux"):
            provide_aux_lines = True
        elif o in ("-v", "--verbose"):
            verbose = True
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

    if pano_width is None:
        pano_width = width

    if pano_height is None:
        pano_height = height

    if verbose:
        # print("input_files:", input_files)
        print("target width:  ", pano_width)
        print("target height: ", pano_height)
        print("with aux lines:", ("true" if provide_aux_lines else "false"))
        print("verbose mode:  ", "true")

    x_img, y_img = pano2omni(pano_width, pano_height, width, height)
    x_img = x_img.astype(np.float32)
    y_img = y_img.astype(np.float32)

    num_files = len(input_files)
    num_threads = 16
    pool = mp.Pool(num_threads)
    num_files_per_thread = int(math.ceil(num_files / num_threads))
    all_args = []
    for i in range(num_threads):
        thread_files = input_files[i * num_files_per_thread: (i + 1) * num_files_per_thread]
        arguments = [thread_files, outdir, width, height, x_img, y_img, provide_aux_lines]
        all_args.append(arguments)
    pool.map(run, all_args)
    pool.close()

    if verbose:
        print()
        print("Finished")


if __name__ == "__main__":
    main()
