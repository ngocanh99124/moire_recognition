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
    padded = cv2.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes)  # Add to the expanded another plane with zeros

    cv2.dft(complexI, complexI)  # this way the result may fit in the source matrix

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
    magII = cv2.normalize(magI, None, 0, 1, cv2.NORM_MINMAX)  # Transform the matrix with float values into a

    if debug == 1:
        cv2.imshow("fourier", magII)

    return magI



def calc(img):
    p = []
    for i in range(0, 256, 1):
        p.append(0.)
    rows, cols = img.shape
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            p[img[i][j]]+=1.
    t_0 = 0.
    t_1 = 0.
    p_0 = 0.
    p_1 = 0.
    rows = rows*1.
    cols = cols*1.
    for i in range(0, 255, 1):
        if p[i] == 0:
            continue
        p[i] = p[i] *1.
        p[i] = p[i]/(rows*cols)
        p_0 = p_0 + p[i]
        p_1 = p_1 + i * p[i]
    avg = -100000
    remember = 0
    for i in range(0, 256, 1):
        p_0 -= p[i]
        p_1 -= p[i]*i
        t_0 += p[i]
        t_1 += p[i]*i
        if p_0 == 0:
            continue
        m1 = p_1 / p_0
        if t_0 == 0:
            continue
        m0 = t_1 / t_0
        eA = t_1 + p_1
        eB = m0*t_0 + m1*p_0
        eAB = m0*t_1 + m1*p_1
        eBB = m0*m0*t_0 + m1*m1*p_0
        p_AB1 = eAB-eA*eB
        p_AB2 = eBB-eB*eB
        if p_AB2 == 0:
            remember = i
            break
        p_AB = p_AB1*p_AB1/p_AB2
        if p_AB > avg:
            remember = i
            avg = p_AB
    peaks = remember
    rows, cols = img.shape
    res = 0
    for i in range(0, rows - 1, 1):
        for j in range(0, cols - 1, 1):
            if img[i][j] > peaks:
                res += 1
    return res / (rows * cols)

def calc_peak(img, peaks):
    rows, cols = img.shape
    res = 0
    for i in range(0, rows - 1, 1):
        for j in range(0, cols - 1, 1):
            if img[i][j] > peaks:
                res += 1
    return res/(rows*cols)

def get_filted(img, k, sigma):
    filted = difference_of_gaussians(img, sigma, k*sigma)
    filter_img = filted*window('hann', img.shape)
    result = fftshift(np.abs(fftn(filter_img)))
    result = cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)
    result = np.uint8(result*255.)
    return result

# def is_spoofing( sigmaMax, k, I):
#     delta = 0.2
#     sigma0 = 0.1
#     while (True):
#         if sigma0 > sigmaMax:
#             return False
#             break
#         a = get_filted(I, k, sigma0)
#         t = calc(a)
#         thres = calc_peak(a, t)
#         if thres < 0.002:
#             #print(thres, t)
#             return True
#             break
#         sigma0 += delta

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
__kernel void check(int row, int col,
    __global const int *img, __global  float * thres)
{
    float p[256];
    // int row = get_global_id(0) col = get_global_id(1);
  
    for (int i = 0; i <= 255; i++) p[i] = 0;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            p[*(img+i*col+j)]+=1;
    float t_0 = 0, t_1 = 0, p_0 = 0, p_1 = 0, m1, m0, eA, eB, eAB, eBB, pAB1, pAB2, pAB;
    for (int i = 0; i <= 255; i++)
    {
        if (p[i] == 0)
            continue;
        p[i] = p[i]/(row*col);
        p_0 = p_0 + p[i];
        p_1 = p_1 + i * p[i];
    }
    float avg = -10000;
    int remember = 0;
    for (int i = 0; i <= 255; i++)
    {
        p_0 -= p[i];
        p_1 -= p[i]*i;
        t_0 += p[i];
        t_1 += p[i]*i;
        if (p_0 == 0)
            continue;
        m1 = p_1/p_0;
        if (t_0 == 0)
            continue;
        m0 = t_1/t_0;
        eA = t_1 + p_1;
        eB = m0*t_0 + m1*p_0;
        eAB = m0*t_1 + m1*p_1;
        eBB = m0*m0*t_0 + m1*m1*p_0;
        pAB1 = eAB-eA*eB;
        pAB2 = eBB-eB*eB;
        if (pAB2 == 0)
        {
            remember = i;
            break;
        }
        pAB = pAB1*pAB1/pAB2;
        if (pAB > avg)
        {
            remember = i;
            avg = pAB;
        }
    }
    float res = 0; int peaks = remember;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        if (*(img+i*col+j) > peaks)
            res+=1;
    float a = row;
    *thres = res/(a*col);

}
""").build()

parser = argparse.ArgumentParser()
parser.add_argument("config_folder", help="The input image file.")
args = parser.parse_args()
config_folder = args.config_folder
folders = open(config_folder, "r")
data = folders.read().split("\n")
print(data)
folder_int = data[0]
folder_out = data[1]
sigmaMax = (float)(data[2])
k = (float)(data[3])
output = open(folder_out, "w+")
print(folder_int)
file_images = os.listdir(folder_int)
for f in tqdm(file_images):
    link_image = os.path.join(folder_int, f)
    img = cv2.imread(link_image, 0)
    if img is None:
        print("can't read image")
    else:
        img = get_filted(img, k, sigmaMax)
        dd = 0
        dem = 0
        rows, cols = img.shape
        queue_ = []
        queue_g = []
        for row in range(0, rows, 400):
            if dd == 1:
                break
            for col in range(0, cols, 400):
                row1 = min(row+400, rows-1)
                col1 = min(col+400, cols-1)
                np_img = np.array(img[row:row1, col:col1], dtype = np.int32)
                res = np.float32(0.)

                a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=np_img)
                res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)
                r, c = np_img.shape
                we = prg.check(queue,np_img.shape, None, np.int32(r), np.int32(c), a_g, res_g)
                res_np = np.empty_like(res)
                queue_.append(res_np)
                cl.enqueue_copy(queue, queue_[dem], res_g)
                dem+=1
                print(res_np, f)
                if res_np < 0.0019:
                    output.write(f)
                    output.write("\n")
                    dd = 1
                    break



