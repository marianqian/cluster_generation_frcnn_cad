import pathlib
import numpy as np
from PIL import Image
from skimage import exposure

def preprocess(path, other=400000, lower=25, upper=100):
    '''
    Returns processed image after MC-GPU simulation.
    
    Arguments:
        path (String or PoSix path) - path to .RAW image including image name.
        other - threshold for maximum pixel value image will keep. Can change, default = 400000. 
        lower - lower threshold percentile of pixel values to keep for image. Can change, default = 25
        upper - higher threshold percentile of pixel values to keep for image. Can change, default = 100. 
    Returns:
        test (ndarray) - processed image.
    '''
    path = pathlib.Path(path)
    raw = np.fromfile(path, dtype='float32', sep="")

    image_size = [2]  # Assumes that there are always 2 images in the RAW image! 
    #2 images in MC-GPU mammogram (1st - primary&scattered xrays, 2nd - primary only)

    image_size.extend((1500, 3000))
    raw = raw.reshape(image_size)

    im_temp = raw[0, :, :] 
    #im_temp = np.rot90(im_temp)  # Rotate the image

    # Finds locations where we want to look at. If values are greater than other (400000), 
    #array is replced with False, if not, then replaced with True which are locations we want to look at. 
    im_mask= np.greater(im_temp, other)
    
    #If false, then replaced with 0.
    im_temp = np.multiply(im_temp, im_mask)  
    
    # Invert the image
    im_temp = np.max(im_temp) - im_temp  
    im_temp = im_temp + 1
    
    # Take the log of the image
    im_temp = np.log(im_temp)  

    #Get minimum where it is greater than 14 - picks smallest value that is greater than 14 
    im_min = np.min(im_temp[im_temp > 14])

    # Perform a shift such that the im_min is 0
    im_temp = im_temp - im_min
    im_temp[im_temp < 0] = 0

    # Zero out the outside of the images again
    im_temp = np.multiply(im_temp, im_mask)
    
    #Zeros out pixels outside of the breast
    im_temp[im_temp == np.max(im_temp)] = 0 
    
    #The value 0.7 can also be changed, but I used 0.7 for all of my pre-processing. 
    #Choosing to keep only the pixel values greater than 0.7 and keeping the 25th to 100th percentile values.
    p1, p2 = np.percentile(im_temp[im_temp > 0.7], (lower, upper))

    test = exposure.rescale_intensity(im_temp, in_range=(p1, p2)) * 255  # Edit this value to change the image bits
    return test

# ======================================================================

def write_out(test, img_name, path='', add=''):
    '''
    Saves the processed image. 
    Arguments:
        test (ndarray) - processed image.
        img_name - file name of image with brackets included. Ex. prj_30mm_2_cluster_malignant_only_142_10nm_s{}.raw.gz.raw
        path - path to folder images are saved in. 
        add (String)- information to add to the file name of the image such as percentile values and mask values etc.  
    '''
    #Saving to a RAW file
    test.astype('uint8').tofile(path+img_name.format(add)) 
    
    img = Image.frombytes('L', (3000, 1500), test.astype('uint8'))
    
    #Saving to a PNG file.
    img.save(path+img_name.format(add)+'.png') 