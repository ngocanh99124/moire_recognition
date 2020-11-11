import cv2
import numpy as np
from skimage.filters import difference_of_gaussians, window
from scipy.fftpack import fftn, fftshift
import argparse
import random, os
from tqdm import tqdm
import pyopencl as cl
import pyopencl.array as cla


def moire_image(I, debug=1):
    rows, cols = I.shape
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)
    padded = cv2.copyMakeBorder(I, 0, m - rows, 0, n - cols,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes)  # Add to the expanded another plane with zeros

    cv2.dft(complexI,
            complexI)  # this way the result may fit in the source matrix

    cv2.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv2.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    magI = planes[0]

    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI)  # switch to logarithmic scale
    cv2.log(magI, magI)

    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows / 2)
    cy = int(magI_cols / 2)
    q0 = magI[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
    q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
    q3 = magI[cx:cx + cx, cy:cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    # print(magI)
    magII = cv2.normalize(magI, None, 0, 1,
                          cv2.NORM_MINMAX)  # Transform the matrix with float values into a

    if debug == 1:
        cv2.imshow("fourier", magII)

    return magI


def get_filted(img, k, sigma):
    filted = difference_of_gaussians(img, sigma, k * sigma)
    filter_img = filted * window('hann', img.shape)
    result = fftshift(np.abs(fftn(filter_img)))
    result = cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)
    result = np.uint8(result * 255.)
    return result

def check(p, l):
    t_0 = 0.
    t_1 = 0.
    p_0 = 0.
    p_1 = 0.
    sl = []
    # print(p, l)
    p = np.array(p, dtype = np.float32)
    for i in range(0, 255, 1):
        sl.append(p[i]*1.)
        if p[i] == 0:
            continue
        p[i] = p[i] * 1.
        p[i] = p[i] / l
        p_0 = p_0 + p[i]
        p_1 = p_1 + i * p[i]
    avg = -100000
    # print("avg", p)
    remember = 0
    for i in range(0, 256, 1):
        p_0 -= p[i]
        p_1 -= p[i] * i
        t_0 += p[i]
        t_1 += p[i] * i
        if p_0 == 0:
            continue
        m1 = p_1 / p_0
        if t_0 == 0:
            continue
        m0 = t_1 / t_0
        eA = t_1 + p_1
        eB = m0 * t_0 + m1 * p_0
        eAB = m0 * t_1 + m1 * p_1
        eBB = m0 * m0 * t_0 + m1 * m1 * p_0
        p_AB1 = eAB - eA * eB
        p_AB2 = eBB - eB * eB
        if p_AB2 == 0:
            remember = i
            break
        p_AB = p_AB1 * p_AB1 / p_AB2
        if p_AB > avg:
            remember = i
            avg = p_AB
    res = 0.0
    # print(remember, sl)
    for i in range(remember, 255, 1):
        res += sl[i]
    # print(res)
    return res/l

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, """
        __kernel void check(__global const int *img, __global  int * thres, int l)
        {
            int gid = get_global_id(0);
            if (gid == 0)
            {
                for (int i = 0; i < l; i++)
                    thres[img[i]] += 1;
            }
        }
        """).build()

def is_moire(img, sigma):
    thres = np.zeros(256, dtype=np.int32)
    thres_g = cl.Buffer(ctx, mf.WRITE_ONLY, thres.nbytes)
    img = get_filted(img, k, sigma)
    np_ar = np.array(img, dtype=np.int32)
    r, c = np_ar.shape
    shape_ = r * c
    np_ar = np.reshape(np_ar, r * c)
    # print("shape", shape_)
    # while shape_%size != 0:
    #     np_ar = np.append(np_ar, [-1])
    #     shape_+=1
    np_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                     hostbuf=np_ar)

    we = prg.check(queue, (1,), None, np_g, thres_g, np.int32(shape_))
    res_np = np.empty_like(thres)
    cl.enqueue_copy(queue, res_np, thres_g, wait_for=[we])
    thres = check(res_np, shape_)

    return thres

parser = argparse.ArgumentParser()
parser.add_argument("config_folder", help="The input image file.")
args = parser.parse_args()
config_folder = args.config_folder
folders = open(config_folder, "r")
data = folders.read().split("\n")
print(data)
folder_int = data[0]
folder_out = data[1]
sigma_ = 0.1
sigmaMax = (float)(data[2])
k = (float)(data[3])
output = open(folder_out, "w+")
print(folder_int)
file_images = os.listdir(folder_int)
print(file_images)
delta = 0.2
size = 1
dem = 0
d2 = 0

for f in tqdm(file_images):
    link_image = os.path.join(folder_int, f)
    img = cv2.imread(link_image, 0)
    print(link_image)
    if img is None:
        print("can't read image")
    else:
        sigma = sigma_
        min_thres = 1
        dd_img = False
        rows, cols = img.shape
        size_r = rows//100
        size_c = cols//100
        if dd_img:
            break
        if size_r < 3 or size_c < 3:
            if not (dd_img):
                thres = is_moire(img, sigma)
                if min_thres > thres:
                    min_thres = thres
                if (thres < 0.001):
                    dem += 1
                    output.write(
                            f + " " + str(thres) + " " + str(img.shape) + "\n")
                    dd_img = True
                    break
                sigma += delta
                continue
        slide_r = rows//size_r
        slide_c = cols//size_c
        print(img.shape, slide_r, slide_c)
        for i in range(0, size_r - 3, 1):
            if dd_img:
                break
            for j in range(0, size_c - 3, 1):
                r = slide_r * i
                rr = slide_r * (i + 2)
                c = slide_c * j
                cc = slide_c * (j + 2)
                thres = is_moire(img[r:rr, c:cc], sigma)
                if min_thres > thres:
                    min_thres = thres
                if (thres < 0.001):
                    dem += 1
                    output.write(
                            f + " " + str(thres) + " " + str(img.shape) + "\n")
                    dd_img = True
                    break
        if not (dd_img):
            output.write(f + " " + str(min_thres) + " " + str(img.shape) + "\n")
        sigma += delta


print(dem)
output.write(str(dem*100 / len(file_images)))



