import os
import numpy as np
import json
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt


def angle(v1, v2, vecmag):
    return(np.arccos(np.sum(v1*v2/vecmag)))


def detect_red_light(imarr, kerndir, imname=False):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    #Remove any Possible 0 pixels
    testim = np.clip(imarr[:, :, :3], 1/255, 255).astype('float32')
    #Normalize pixels to magnitude 1
    testim = np.divide(testim, np.expand_dims(
        np.linalg.norm(testim, axis=-1), axis=-1))
    detmvals = []
    detbbs = []
    kerns = glob(kerndir + '/*.png')
    #Read in prototypes
    for kernname in kerns:
        kern = np.clip(np.asarray(Image.open(kernname))[
                       :, :, :3], 1/255, 255).astype('float32')
        kern = np.divide(kern, np.expand_dims(
            np.linalg.norm(kern, axis=-1), axis=-1))
        kernx, kerny, kernz = kern.shape
        flatkern = kern.flatten()
        outim = np.zeros(testim.shape[:2])
        #Constant scalar which varies by size of prototype
        denominator = np.linalg.norm(kern.flatten())**2
        #Step by 2 pixels for speed increase
        for xval in np.arange(0, testim.shape[0]-kernx-1, 3):
            for yval in np.arange(0, testim.shape[1]-kerny-1, 3):
                extract = testim[xval:xval+kernx, yval:yval+kerny, :kernz]
                outval = np.cos(
                    angle(flatkern, extract.flatten(), denominator))
                outim[xval, yval] = outval
        #Iteratively find maximum and remove areas of high score to allow multiple matches with same prototype
        for nit in range(100):
            if np.max(outim) >= .975:
                goodinds = np.where(outim == np.max(outim))
                goodrow, goodcol = goodinds
                goodrow, goodcol = goodrow[0], goodcol[0]
                boundbox = [goodrow, goodcol, goodrow+kernx, goodcol+kerny]
                detmvals.append(np.max(outim))
                detbbs.append(boundbox)
                outim[goodrow-10:goodrow+10, goodcol-10:goodcol+10] = 0
            else:
                break
    badinds = []
    #Remove duplicate detection and maintain the highest score in overlapping sets
    for bbind in range(len(detbbs)):
        thisbb = np.array(detbbs[bbind])
        diffs = np.array([np.linalg.norm(thisbb[:2] - x[:2]) for x in detbbs])
        if len(diffs[diffs < 24]) > 0:
            initdupinds = np.where(diffs < 24)[0]
            bestind = initdupinds[np.argmax(np.array(detmvals)[initdupinds])]
            badinds.extend([x for x in initdupinds if x != bestind])
    badinds = list(set(badinds))
    for ele in sorted(badinds, reverse=True):
        del detbbs[ele]
    if imname:
        print('Finished ' + imname)
    return(detbbs)


def gen_sample_ims(fnames, kerndir):
    if not os.path.exists('figures/'):
        os.makedirs('figures')
    for fnm in fnames:
        jpgname = str.split(fnames, '.jpg')[0][-6:]
        imarr = np.asarray(Image.open(fnm))
        detbbs = detect_red_light(imarr, kerndir)
        plt.imshow(imarr/255)
        for boundbox in detbbs:
            plt.scatter([boundbox[1], boundbox[3]], [boundbox[0],
                        boundbox[2]], marker='x', color='blue', s=15)
        plt.title(str.split(fnm, '.')[0][-6:])
        plt.savefig('figures/'+jpgname+'_predimage.png', dpi=300)


## Commented code below generates the sample figures
# fnames=['data/RedLights2011_Medium/RL-002.jpg' \
# 'data/RedLights2011_Medium/RL-094.jpg' \
# 'data/RedLights2011_Medium/RL-115.jpg' \
# 'data/RedLights2011_Medium/RL-268.jpg' \
# 'data/RedLights2011_Medium/RL-334.jpg'] \
# kerndir='data/kernels_noproc_nohousing'
# gen_sample_ims(fnames, kerndir)


# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = 'data/hw01_preds'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(
        imarr=I, kerndir='data/kernels_noproc_nohousing', imname=file_names[i])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
    json.dump(preds, f)
