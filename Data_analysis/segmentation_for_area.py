import numpy as np
import matplotlib.pyplot as plt
import tifffile
import matplotlib.animation as animation
from scipy import ndimage as ndi
from skimage.morphology import  binary_erosion, binary_dilation
import pandas as pd
from copy import deepcopy
from skimage.feature import canny
import cv2

filename = "C:\\example.tif"
frames = tifffile.imread(filename)

def largest_feature(binary_image=None):
	''' Find the largest contiguous feature in a binary image.
	Keyword arguments:
	binary_image (bool 2D array) -- A binary image with some number of clusters
	Return arguments:
	max_feature (bool 2D array) -- A new binary image containing only the largest cluster
	labeled_image (int 2D array) -- A labeled version of the input image, where the pixels of each cluster
										have been given an integer value indicating their grouping.
	'''
	labeled_image, num_features = ndi.label(binary_image) # Label connected components in the binary image
	label_indexes = range(1, num_features + 1)
	labeled_feature_sizes = ndi.sum(binary_image, labels=labeled_image, index = label_indexes)
	largest_feature_label = np.where(labeled_feature_sizes==np.max(labeled_feature_sizes))[0][0]
	largest_feature_mask = np.zeros_like(labeled_image)
	largest_feature_mask[labeled_image==largest_feature_label+1] = 1
	return largest_feature_mask, labeled_image


def clean(data,scale,post_inoculation_t):
    ''' Process greyscale epifluorescence images by: removing uneven illumination artifacts with homomorphic filter, identifying the edges of the objects pictured, filling gaps in the outline, and filling in whole identified object (biofilm).
    Keyword arguments:
    data (int 2D array) -- A greyscale image
    scale (int) -- Scale of image pixels (ex. 1 micron/pixel)
    post_inoculation_t (int) -- Time between inoculating sample and data collection (in the case of taking timelapse images of biofilm growth)
    '''
    time = np.arange(0.25+post_inoculation_t,len(frames)/4+0.25+post_inoculation_t,0.25)

    cleaned = []
    area = np.zeros((len(data)))
    for t in range(0, len(data)):
        img = data[t]
        m_xy = deepcopy(img).astype('float')
        m_xy -= np.min(img)
        m_xy *= (1.0/np.max(m_xy))
        # avoid ln(0), renorm later
        m_xy+=1
        # take ln of image
        img_log = np.log(m_xy, dtype=np.float64)

        # do dft saving as complex output
        dft = np.fft.fft2(img_log)

        # cv2.imshow('dft',prepare_spectrum(dft))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # create black circle on white background for high pass filter
        radius = 100
        mask = np.zeros_like(img, dtype=np.float64)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        mask = cv2.circle(mask, (cx,cy), radius, 1, -1)
        mask = 1 - mask

        # antialias mask via blurring
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        # mask = cv2.GaussianBlur(mask, (100,100), 0)

        # apply mask to dft_shift
        dft_shift_filtered = np.multiply(dft_shift,mask)

        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift_filtered)

        # do idft saving as complex
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        img_back = np.abs(img_back)

        # apply exp to reverse the earlier log
        img_homomorphic = np.exp(img_back, dtype=np.float64)

        # scale result
        img_homomorphic -= np.min(img_homomorphic)
        img_homomorphic *= (1.0/np.max(img_homomorphic))
        
        # identify edges of result
        canny_vers = canny(img_homomorphic)

        # fill in gaps in edges and fill in whole object
        dil = binary_dilation(canny_vers,footprint=np.ones((30,30))).astype(int)
        fill = ndi.binary_fill_holes(dil).astype(bool)
        subject = largest_feature(fill)[0]
        ero = binary_erosion(subject,footprint=np.ones((30,30)))

        # overlay image with previous frame to fill in any mistakes (areas that were previously filled in that were misidentified in the following frame)
        if t > 0:
            final_img = ero + cleaned[t-1]
        else:
            final_img = ero

        # redo some of the cleanup if any of the object touches the edges of the frame (which I know my objects should not do)
        if np.any(final_img[::len(final_img[0])-1,:] != 0) or np.any(final_img[:,::len(final_img[1])-1] != 0):
            ero2 = binary_erosion(ero,footprint=np.ones((10,10)))
            largest_feature2 = largest_feature(ero2)[0]
            dil2 = binary_dilation(largest_feature2,footprint=np.ones((10,10)))
            final_img = dil2

        cleaned.append(final_img)
        
        area[t] = np.sqrt(scale**2*(np.sum(final_img[t])/255))
        
    image_stack = np.stack(cleaned, axis=0) # Stack images along a new axis
    return(time,image_stack,area)

output = clean(frames)
time = output[0]
cleaned = output[1]
area = output[2]

## check for individual frame
fig, ax = plt.subplots()
ax.clear()
ax.set_title(f"Cleaned data - Time: {time[0]}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.imshow(cleaned, cmap=plt.cm.Blues, alpha=1, origin='lower',aspect='auto')
plt.show()

# plot entire output
fig, ax = plt.subplots()
def update(frame):
    ax.clear()
    ax.set_title(f"Cleaned data - Time: {time[frame]}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.imshow(cleaned[frame], cmap=plt.cm.Blues, alpha=1, origin='lower',aspect='auto')

ani = animation.FuncAnimation(fig, update, frames=len(time), interval=1000)
plt.show()
writer = animation.PillowWriter(fps=1)
ani.save('Figures/Example.gif', writer=writer)
    
# plot area
fig,ax = plt.subplots(figsize=(20,20))
ax.set_xlabel("time (hpi)")
ax.set_ylabel("√Area microns")
ax.set_title(f"Example")
ax.plot(time,area,label=r"Example")
fig.legend()
fig.savefig(f"Figures/Example.png")
plt.close()


# save area as csv
data = {'Time':time,'Area':area}
df = pd.DataFrame(data)
df.to_csv(f"Raw_data\Example.csv")