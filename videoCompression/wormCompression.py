# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:19:58 2015

@author: ajaver
"""
import numpy as np
import cv2
import h5py

from readVideoffmpeg import readVideoffmpeg
from parallelProcHelper import sendQueueOrPrint, timeCounterStr
from image_difference_mask import image_difference_mask

def getROIMask(image,  min_area = 100, max_area = 5000):
    '''
    Calculate a binary mask to mark areas where it is possible to find worms.
    Objects with less than min_area or more than max_area pixels are rejected.
    '''
    #Objects that touch the limit of the image are removed. I use -2 because openCV findCountours remove the border pixels
    IM_LIMX = image.shape[0]-2
    IM_LIMY = image.shape[1]-2
    
    #adaptative threshold is the best way to find possible worms. I setup the parameters manually, they seems to work fine if there is no condensation in the sample
    mask = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,15)

    #find the contour of the connected objects (much faster than labeled images)
    [contours, hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #find good contours: between max_area and min_area, and do not touch the image border
    goodIndex = []
    for ii, contour in enumerate(contours):
        if np.all(contour!=1) and np.all(contour[:,:,0] !=  IM_LIMX)\
        and np.all(contour[:,:,1] != IM_LIMY):
            area = cv2.contourArea(contour)
            if (area>=min_area) and (area<=max_area):
                goodIndex.append(ii)
    
    #typically there are more bad contours therefore it is cheaper to draw only the valid contours
    mask = np.zeros(image.shape, dtype=image.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 255, cv2.cv.CV_FILLED)
    
    #drawContours left an extra line if the blob touches the border. It is necessary to remove it
    mask[0,:] = 0; mask[:,0] = 0; mask[-1,:] = 0; mask[:,-1]=0;
    
    #dilate the elements to increase the ROI, in case we are missing something important
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) 
    mask = cv2.dilate(mask, struct_element, iterations = 3)
    
    #the gecko images have a time stamp in the image border
    cv2.rectangle(mask, (0,0), (479,15), 255, thickness=-1) 

    return mask
   
def compressVideo(video_file, masked_image_file, buffer_size = 25, \
save_full_interval = 5000, max_frame = 1e32, base_name = '', \
status_queue = '', check_empty = True,  useVideoCapture = True):
    '''
    Compressed video in "video_file" by selecting ROI and making the rest of 
    the image zero (creating a large amount of redundant data)
    the final images are saving in the file given by "masked_image_file" 
    as hdf5 with gzip compression
    To reduce the processing load buffer_size images are collected, and a minimum filter 
    applied over the stack. In this way the pixels corresponding to the worms are 
    preserved as black pixels in the min-average image, and only in this 
     image the the binary mask with possible worms is calculated.

     MAX_N_PROCESSES -- number of processes using during image processing, if -1, this value is set to the number of cpu is using
     save_full_interval --  Full frame is saved every 'save_full_interval' in '/full_data'
     max_frame -- maximum number of frames to be analyzed. Set this value to a large value to compress all the video    
     status_queue -- queue were the status is sended. Only used in multiprocessing case 
     base_name -- processes identifier. Only used in the multiprocessing case.
    '''
    
    #max_frame = 1000
    #open video to read
    if not useVideoCapture:
        vid = readVideoffmpeg(video_file);
        im_height = vid.height;
        im_width = vid.width;
    else:
        vid = cv2.VideoCapture(video_file);
        im_width= vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        im_height= vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    
    if check_empty:
        rand_ind = np.random.randint(0, im_height*im_width-1, (200)); #select 200 random index use to check if the image is empty
    
    
    #open hdf5 to store the processed data
    mask_fid = h5py.File(masked_image_file, "w");
    #open node to store the compressed (masked) data
    mask_dataset = mask_fid.create_dataset("/mask", (0, im_height,im_width), 
                                    dtype = "u1", maxshape = (None, im_height,im_width), 
                                    chunks = (1, im_height,im_width),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
    #full frames are saved in "/full_data" every save_full_interval frames
    full_dataset = mask_fid.create_dataset("/full_data", (0, im_height,im_width), 
                                    dtype = "u1", maxshape = (None, im_height,im_width), 
                                    chunks = (1, im_height,im_width),
                                    compression="gzip", 
                                    compression_opts=9,
                                    shuffle=True);
    full_dataset.attrs['save_interval'] = save_full_interval
    
    im_diff_set = mask_fid.create_dataset('/im_diff', (0,), 
                                          dtype = 'f4', maxshape = (None,), 
                                        chunks = True, compression = "gzip", compression_opts=9, shuffle = True)
    

    
    #intialize frame number
    frame_number = 0;
    full_frame_number = 0;
    image_prev = np.zeros([]);
    
    #initialize timers
    progressTime = timeCounterStr('Compressing video.');

    while frame_number < max_frame:
        ret, image = vid.read() #get video frame, stop program when no frame is retrive (end of file)
        if ret == 0:
            break
        
        if useVideoCapture:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        #check if the 200 randomly selected index are equal (much faster that check the whole image)
        if check_empty and np.all(image.flat[rand_ind]-np.mean(image.flat[rand_ind])==0):
            continue
        
        frame_number += 1;
        
        #import matplotlib.pylab as plt
        #plt.figure()
        #plt.imshow(image)
        
        #Resize mask array every 1000 frames (doing this every frame does not impact much the performance)
        if (frame_number)%1000 == 1:
            mask_dataset.resize(frame_number + 1000, axis=0); 
            im_diff_set.resize(frame_number + 1000, axis=0); 
        #Add a full frame every save_full_interval
        if frame_number % save_full_interval== 1:
            
            full_dataset.resize(full_frame_number+1, axis=0); 
            assert(np.floor(frame_number/save_full_interval) == full_frame_number) #just to be sure that the index we are saving in is what we what we are expecting
            full_dataset[full_frame_number,:,:] = image.copy()
            full_frame_number += 1;

        
        ind_buff = (frame_number-1) % buffer_size #buffer index
        
        #initialize the buffer when the index correspond to 0
        if ind_buff == 0:
            Ibuff = np.zeros((buffer_size, im_height, im_width), dtype = np.uint8)

        #add image to the buffer
        Ibuff[ind_buff, :, :] = image.copy()
        
        if ind_buff == buffer_size-1:
            #calculate the mask only when the buffer is full
            mask = getROIMask(np.min(Ibuff, axis=0))
            
            #mask all the images in the buffer
            for ii in range(Ibuff.shape[0]):
                #create a reference copy
                im = Ibuff[ii,:,:]; 
                #bitwise_and by reference (keep values having 255 in the mask)
                cv2.bitwise_and(im,mask, im); 

            #add buffer to the hdf5 file
            mask_dataset[(frame_number-buffer_size):frame_number,:,:] = Ibuff
        
        #calculate difference between image (it's usefull to indentified corrupted frames)
        image[0:15,0:479] = 0; #remove timestamp before calculation
        if image_prev.shape:
            im_diff_set[frame_number] = image_difference_mask(image,image_prev)
        image_prev = image.copy();  
        
        if frame_number%25 == 0:
            #calculate the progress and put it in a string
            progress_str = progressTime.getStr(frame_number)
            sendQueueOrPrint(status_queue, progress_str, base_name);
            
            
    
    if mask_dataset.shape[0] != frame_number:
        mask_dataset.resize(frame_number, axis=0);
        im_diff_set.resize(frame_number, axis=0);
        
    if full_dataset.shape[0] != full_frame_number:
        full_dataset.resize(full_frame_number, axis=0);
        
    #close the video and hdf5 files
    vid.release() 
    mask_fid.close()
    sendQueueOrPrint(status_queue, 'Compressed video done.', base_name);

