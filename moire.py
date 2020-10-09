import cv2
import numpy as np
from skimage.filters import difference_of_gaussians, window
from scipy.fftpack import fftn, fftshift
import argparse
import random, os

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

    #if debug == 1:
    #    cv2.imshow("fourier", magII)

    return magI

def m_0(t, p):
    res = 0.
    res1 = 0.
    for i in range(0, t+1, 1):
        if i in p:
            res += i*p[i]
            res1 = res1 + p[i]
    res = res/res1
    return res
def m_1(t, p):
    res = 0.
    res1 = 0.
    for i in range(t+1, 255, 1):
        if i in p:
            res = res + i * p[i]
            res1 = res1 + p[i]
    res = res / res1
    return res
def e_AB(t, p):
    res = 0.
    for i in range(0, t-1, 1):
        if i in p:
            res = res + i * p[i]

    res1 = res * m_0(t, p)
    res = 0.
    for i in range(t+1, 255, 1):
        if i in p:
            res = res + i * p[i]
    res1 += res *m_1(t, p)
    return res1

def e_A(p):
    res = 0.
    for i in range(0, 255, 1):
        if i in p:
            res = res + i * p[i]
    return res

def e_B(t, p):
    res = 0.
    for i in range(0, t, 1):
        if i in p:
            res = res + p[i]
    res1 = res*m_0(t, p)
    res = 0.
    for i in range(t+1, 255,1):
        if i in p:
            res = res + p[i]
    res1 += res * m_1(t, p)
    return res1

def e_BB(t, p):
    res = 0.
    for i in range(0, t, 1):
        if i in p:
            res = res + p[i]
    M0 = m_0(t, p)
    M1 = m_1(t, p)
    res1 = res * M0*M0
    res = 0.
    for i in range(t+1, 255, 1):
        if i in p:
            res = res + p[i]
    res1+=res*M1*M1
    return res1

def calc(img):
    p = {}
    rows, cols = img.shape
    for i in range(0, rows-1, 1):
        for j in range(0, cols-1, 1):
            if img[i][j] in p:
                p[img[i][j]]+=1
            else:
                p[img[i][j]]=1
    for i in range(0, 255, 1):
        if i in p:
            p[i] = p[i]/(rows*cols)
    #print(p)
    avg = -100000
    remember = 0
    for t in range(0, 255, 1):
        if t in p:
            try:
                eB = e_B(t, p)
                p_AB1= e_AB(t, p)-e_A(p)*e_B(t, p)
                p_AB2= (e_BB(t, p)-eB*eB)
                #print(p_AB1*p_AB1, p_AB2,t)
                p_AB = p_AB1*p_AB1/p_AB2
                if p_AB > avg:
                    avg = p_AB
                    remember = t
                    #print(p_AB, t)
            except:
                #print("math err")
                a = 0

    return remember

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
    #print("result", result)
    result = np.uint8(result*255.)
    #cv2.imwrite("result/log"+str(sigma)+".jpg", result)
    return result

def show_thres(img, thres, sigma, debug=1):
    rows, cols = img.shape
    img = np.zeros([rows, cols])
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            if img[i][j] > thres:
                img[i][j] = 255
            else:
                img[i][j] = 0

    #if debug == 1:
    #    cv2.imwrite("result/thes"+str(sigma)+".jpg",img)

def is_spoofing(I):
    delta = 0.2
    sigma0 = 0.1
    sigmaMax = 0.5
    k = 3
    while (True):
        if sigma0 > sigmaMax:
            return False
            break
        # a = moire_image(I, 1)
        a = get_filted(I, k, sigma0)
        t = calc(a)
        #print(t)
        thres = calc_peak(a, t)
        show_thres(a, t, sigma0, 1)
        if thres < 0.001:
            #print(thres, t)
            return True
            break
        sigma0 += delta

parser = argparse.ArgumentParser()
parser.add_argument("config_folder", help="The input image file.")
args = parser.parse_args()
config_folder = args.config_folder
folders = open(config_folder, "r")
data = folders.read().split("\n")
folder_int = data[0]
folder_out = data[1]
output = open(folder_out, "w+")
print(folder_int)
file_images = os.listdir(folder_int)
for f in file_images:
    link_image = os.path.join(folder_int, f)
    img = cv2.imread(link_image, 0)
    if img is None:
        print("can't read image")
    else:
        rows, cols = img.shape
        if rows>200 and cols > 200:
            x = random.randrange(0, rows-200, 1)
            y = random.randrange(0, cols-200, 1)
            img1 = img[x:x+199, y:y+199]
            if is_spoofing(img1):
                output.write(f)
                output.write("\n")
            else:
                x = random.randrange(0, rows - 200, 1)
                y = random.randrange(0, cols - 200, 1)
                img1 = img[x:x + 199, y:y + 199]
                if is_spoofing(img1):
                    output.write(f)
                    output.write("\n")

        else:
            if is_spoofing(img):
                output.write(f)
                output.write("\n")
