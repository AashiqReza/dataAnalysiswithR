import numpy as np

import numpy as np
import cv2
from skimage.measure import block_reduce
import math
import time
start_time = time.time()
import numpy as np
import cv2
import os
import cv2
import numpy as np


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def fractal_dimension(Z):
    # Only for 2d image
    assert(len(Z.shape) == 2)
    threshold = Z.mean()

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def lacunarity(image, box_size):
    boxes = np.zeros((box_size, box_size))
    for i in range(box_size):
        for j in range(box_size):
            boxes[i, j] = np.sum(image[i::box_size, j::box_size])

    mean = np.mean(boxes)
    variance = np.var(boxes)
    cv = np.sqrt(variance) / mean
    lambda_value = cv ** 2

    return lambda_value


def hurst_exponent(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Flatten the image into a 1D array
    data = gray.flatten()
    # Calculate the standard deviation of the differences between a series and its lagged version, for a range of possible lags
    std_dev = [np.std(np.subtract(data[lag:], data[:-lag])) for lag in range(2, 20)]
    # Estimate the Hurst exponent as the slope of the log-log plot of the number of lags versus the mentioned standard deviations
    m = np.polyfit(np.log(range(2, 20)), np.log(std_dev), 1)
    hurst = m[0] * 2
    return hurst


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

no = 'Training/no_tumor/'
no_tumor = load_images_from_folder(no)

gli = 'Training/glioma_tumor/'
glioma = load_images_from_folder(gli)
menin = 'Training/meningioma_tumor/'
menin = load_images_from_folder(menin)

pit = 'Training/pituitary_tumor/'
pitui = load_images_from_folder(pit)

#
FF_pit = []
for image in pitui:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_pit.append(image_features)
#
FF_menin = []
for image in menin:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_menin.append(image_features)
#
FF_no = []
for image in no_tumor:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_no.append(image_features)

FF_gli = []
for image in glioma:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_gli.append(image_features)

#
import pandas as pd
FF_pit = pd.DataFrame(FF_pit)
FF_no = pd.DataFrame(FF_no)
FF_gli = pd.DataFrame(FF_gli)
FF_menin = pd.DataFrame(FF_menin)
FF_pit['label'] = 'Pituitary tumor'
FF_no['label'] = 'No Tumor'
FF_gli['label'] = 'Glioma tumor'
FF_menin['label'] = 'Meninglioma tumor'

# FD CALCULATION
from PIL import Image, ImageFilter
import imageio

FD_no = []
folder = 'Training/no_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_no.append(fractal_dimension(I))
    
FD_glio = []
folder = 'Training/glioma_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_glio.append(fractal_dimension(I))

FD_menin = []
folder = 'Training/meningioma_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_menin.append(fractal_dimension(I))
#
FD_pit = []
folder = 'Training/pituitary_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_pit.append(fractal_dimension(I))
#
FD_no = pd.DataFrame(FD_no)
FF_no['FD'] = FD_no[0]

FD_glio = pd.DataFrame(FD_glio)
FF_gli['FD'] = FD_glio[0]

FD_pit = pd.DataFrame(FD_pit)
FF_pit['FD'] = FD_pit[0]

FD_menin = pd.DataFrame(FD_menin)
FF_menin['FD'] = FD_menin[0]

train_data = pd.concat([FF_no, FF_gli, FF_pit, FF_menin], ignore_index=True)

############################################################################## TEST

no = 'Testing/no_tumor/'
no_tumor = load_images_from_folder(no)

gli = 'Testing/glioma_tumor/'
glioma = load_images_from_folder(gli)

menin = 'Testing/meningioma_tumor/'
menin = load_images_from_folder(menin)

pit = 'Testing/pituitary_tumor/'
pitui = load_images_from_folder(pit)

FF_pit = []
for image in pitui:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_pit.append(image_features)

FF_menin = []
for image in menin:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_menin.append(image_features)

FF_no = []
for image in no_tumor:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_no.append(image_features)

FF_gli = []
for image in glioma:
    hu = hurst_exponent(image)
    b = lacunarity(image, 250)
    image_features = {
        'hursdrof': hu,
        'lacunarity': b,
    }
    FF_gli.append(image_features)

import pandas as pd
FF_pit = pd.DataFrame(FF_pit)
FF_no = pd.DataFrame(FF_no)
FF_gli = pd.DataFrame(FF_gli)
FF_menin = pd.DataFrame(FF_menin)
FF_pit['label'] = 'Pituitary tumor'
FF_no['label'] = 'No Tumor'
FF_gli['label'] = 'Glioma tumor'
FF_menin['label'] = 'Meninglioma tumor'

# FD CALCULATION
from PIL import Image, ImageFilter
import imageio

FD_no = []
folder = 'Testing/no_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_no.append(fractal_dimension(I))
    
FD_glio = []
folder = 'Testing/glioma_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_glio.append(fractal_dimension(I))

FD_menin = []
folder = 'Testing/meningioma_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_menin.append(fractal_dimension(I))

FD_pit = []
folder = 'Testing/pituitary_tumor/'
for filename in os.listdir(folder):
    I = imageio.imread(os.path.join(folder,filename))
    I = rgb2gray(I)
    FD_pit.append(fractal_dimension(I))

FD_no = pd.DataFrame(FD_no)
FF_no['FD'] = FD_no[0]

FD_glio = pd.DataFrame(FD_glio)
FF_gli['FD'] = FD_glio[0]

FD_pit = pd.DataFrame(FD_pit)
FF_pit['FD'] = FD_pit[0]

FD_menin = pd.DataFrame(FD_menin)
FF_menin['FD'] = FD_menin[0]

test_data = pd.concat([FF_no, FF_gli, FF_pit, FF_menin], ignore_index=True)


train_data.to_csv('train1.csv', index=False)
test_data.to_csv('test1.csv', index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")