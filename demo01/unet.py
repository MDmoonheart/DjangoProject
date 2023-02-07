from __future__ import print_function
import os
import sys
import numpy as np
from keras.models import Model as Kmodel
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from scipy.ndimage import gaussian_filter
from PIL import Image as im



K.set_image_data_format('channels_last')  # TF dimension ordering in this code
img_rows = int(512/2)
img_cols = int(512/2)
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Kmodel(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

def load_liver_imgs(path,img_rows,img_cols):
    img3D = np.empty(shape=(len(os.listdir(path)),512,512))
    imgs_for_train = np.empty(shape=(len(os.listdir(path)),img_rows,img_cols))
    k=0
    for s in os.listdir(path):
        imfile = os.path.join(path,s)
        dcminfo = pydicom.read_file(imfile)
        rawimg = dcminfo.pixel_array
        img = apply_modality_lut(rawimg,dcminfo)
        img3D[k,:,:] = img
        img = cv2.resize(img, (img_rows,img_cols), interpolation = cv2.INTER_AREA)
        imgs_for_train[k,:,:] = img
        k += 1
    imgs_for_train[imgs_for_train>255]=255
    imgs_for_train[imgs_for_train<0]=0
    return img3D, imgs_for_train

def prepare_liver_test(path,img_rows,img_cols):
    img3D, imgs_for_test = load_liver_imgs(path,img_rows,img_cols)
    imgs_for_test = imgs_for_test.astype('float32')
    return img3D, imgs_for_test

def load_imgs(path,img_rows,img_cols):
    img3D = np.empty(shape=(len(os.listdir(path)),512,512))
    imgs_for_train = np.empty(shape=(len(os.listdir(path)),img_rows,img_cols))
    k=0
    for s in os.listdir(path):
        imfile = os.path.join(path,s)
        dcminfo = pydicom.read_file(imfile)
        rawimg = dcminfo.pixel_array
        img = apply_modality_lut(rawimg,dcminfo)
        img3D[k,:,:] = img
        img = cv2.resize(img, (img_rows,img_cols), interpolation = cv2.INTER_AREA)
        imgs_for_train[k,:,:] = img
        k += 1
        
    offset = 100
    upperl = 255 - offset
    lowerl = 0 - offset
    imgs_for_train[imgs_for_train>upperl]=upperl
    imgs_for_train[imgs_for_train<lowerl]=lowerl
    imgs_for_train += offset
    imgs_for_train *= 1
    return img3D, imgs_for_train

def prepare_test(path,img_rows,img_cols):
    img3D, imgs_for_test = load_imgs(path,img_rows,img_cols)
    imgs_for_test = imgs_for_test.astype('float32')
    return img3D, imgs_for_test

def find_cover_slices(objmask3D):
    objmask3D.astype(int)
    objmasksize = np.zeros(shape=(objmask3D.shape[0]),dtype=int)
    for k in range(objmask3D.shape[0]):
        if k>0:
            if np.sum(np.multiply(objmask3D[k-1,:,:],objmask3D[k,:,:]))>0:
                objmasksize[k] = objmasksize[k-1] + np.sum(objmask3D[k,:,:])
        else:
            objmasksize[k] = np.sum(objmask3D[k,:,:])
    objlastslice = np.argmax(objmasksize)
    zerosize = 0
    objfirstslice = 0
    for k in range(objmask3D.shape[0]):
        k1 = objmask3D.shape[0]-k-1
        if k1 <= objlastslice:
              if objmasksize[k1]==zerosize:
                    objfirstslice = k1+1
                    zerosize = -1
    return objfirstslice,objlastslice

def slice_selection(casefolder_path, save_path):
    # load pre-trained U-Net
    livermodel = get_unet()
    livermodel.load_weights(sys.path[0]+'/model/202101210045weights.h5')
    # load mean and std
    stat_para = np.load(sys.path[0]+'/model/202101202341mean_std.npz')
    liver_mean = stat_para['mean']
    liver_std = stat_para['std']
    print()
    liver_img3D, liver_imgs_test = prepare_liver_test(casefolder_path,img_rows,img_cols)
    for sliceidx in range(liver_imgs_test.shape[0]):
        orig_img = np.copy(liver_imgs_test[sliceidx,:,:])
        liver_imgs_test[sliceidx,:,:] = gaussian_filter(np.float32(orig_img), sigma=2)
    liver_imgs_test = liver_imgs_test[...,np.newaxis]
    liver_imgs_test -= liver_mean
    liver_imgs_test /= liver_std

    liver_mask_test = livermodel.predict(liver_imgs_test, verbose=1)

    maxmaskpx = 0
    msliceidx = 0
    for sliceidx in range(liver_mask_test.shape[0]):
        maskpx = np.count_nonzero(liver_mask_test[sliceidx,:,:,0] == 1)
        if maxmaskpx < maskpx:
            msliceidx = sliceidx
            maxmaskpx = maskpx


    maxliverimg = liver_img3D[msliceidx,:,:]
    maxliverimg = cv2.resize(maxliverimg, (img_rows,img_cols), interpolation = cv2.INTER_AREA)
    offset = 100
    upperl = 255 - offset
    lowerl = 0 - offset
    maxliverimg[maxliverimg>upperl]=upperl
    maxliverimg[maxliverimg<lowerl]=lowerl
    maxliverimg += offset
    maxliverimg *= 1
    maxlivermsk = liver_mask_test[msliceidx,:,:,0].astype('uint8')
    selectedimg = maxliverimg * maxlivermsk
    selectedimg.astype('uint8')

    # segmentation = models.Segmentation(caseID='1',segID=1,seg=selectedimg.tobytes(),is_selected=True)
    # segmentation.save()


    imgfname = os.path.join(save_path, 'Case001' + '.png')
    print(imgfname)
    imgdata = im.fromarray(selectedimg)
    imgdata = imgdata.convert('RGB')
    imgdata.save(imgfname) 
    