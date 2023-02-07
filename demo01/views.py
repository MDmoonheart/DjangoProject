from __future__ import print_function

from django.shortcuts import render,HttpResponse
from . import resnet2 as resnet
from mindspore import Model, Tensor, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
import os
import sys
from keras.models import Model as kModel
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from scipy.ndimage import gaussian_filter
from PIL import Image as im
from . import models
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd



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

    model = kModel(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])

    return model
def load_liver_imgs(path, img_rows, img_cols):
    img3D = np.empty(shape=(len(os.listdir(path)), 512, 512))
    imgs_for_train = np.empty(shape=(len(os.listdir(path)), img_rows, img_cols))
    k = 0
    for s in os.listdir(path):
        imfile = os.path.join(path, s)
        dcminfo = pydicom.read_file(imfile)
        rawimg = dcminfo.pixel_array
        img = apply_modality_lut(rawimg, dcminfo)
        img3D[k, :, :] = img
        img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        imgs_for_train[k, :, :] = img
        k += 1
    imgs_for_train[imgs_for_train > 255] = 255
    imgs_for_train[imgs_for_train < 0] = 0
    return img3D, imgs_for_train
def prepare_liver_test(path, img_rows, img_cols):
    img3D, imgs_for_test = load_liver_imgs(path, img_rows, img_cols)
    imgs_for_test = imgs_for_test.astype('float32')
    return img3D, imgs_for_test
def load_imgs(path, img_rows, img_cols):
    img3D = np.empty(shape=(len(os.listdir(path)), 512, 512))
    imgs_for_train = np.empty(shape=(len(os.listdir(path)), img_rows, img_cols))
    k = 0
    for s in os.listdir(path):
        imfile = os.path.join(path, s)
        dcminfo = pydicom.read_file(imfile)
        rawimg = dcminfo.pixel_array
        img = apply_modality_lut(rawimg, dcminfo)
        img3D[k, :, :] = img
        img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        imgs_for_train[k, :, :] = img
        k += 1

    offset = 100
    upperl = 255 - offset
    lowerl = 0 - offset
    imgs_for_train[imgs_for_train > upperl] = upperl
    imgs_for_train[imgs_for_train < lowerl] = lowerl
    imgs_for_train += offset
    imgs_for_train *= 1
    return img3D, imgs_for_train
def prepare_test(path, img_rows, img_cols):
    img3D, imgs_for_test = load_imgs(path, img_rows, img_cols)
    imgs_for_test = imgs_for_test.astype('float32')
    return img3D, imgs_for_test
def find_cover_slices(objmask3D):
    objmask3D.astype(int)
    objmasksize = np.zeros(shape=(objmask3D.shape[0]), dtype=int)
    for k in range(objmask3D.shape[0]):
        if k > 0:
            if np.sum(np.multiply(objmask3D[k - 1, :, :], objmask3D[k, :, :])) > 0:
                objmasksize[k] = objmasksize[k - 1] + np.sum(objmask3D[k, :, :])
        else:
            objmasksize[k] = np.sum(objmask3D[k, :, :])
    objlastslice = np.argmax(objmasksize)
    zerosize = 0
    objfirstslice = 0
    for k in range(objmask3D.shape[0]):
        k1 = objmask3D.shape[0] - k - 1
        if k1 <= objlastslice:
            if objmasksize[k1] == zerosize:
                objfirstslice = k1 + 1
                zerosize = -1
    return objfirstslice, objlastslice


def slice_selection(casefolder_path, save_path ,caseid):
    # load pre-trained U-Net
    livermodel = get_unet()
    print(sys.path[0] , '/model/202101210045weights.h5')
    livermodel.load_weights(sys.path[0] + '/model/202101210045weights.h5')
    # load mean and std
    stat_para = np.load(sys.path[0] + '/model/202101202341mean_std.npz')
    liver_mean = stat_para['mean']
    liver_std = stat_para['std']

    liver_img3D, liver_imgs_test = prepare_liver_test(casefolder_path, img_rows, img_cols)
    for sliceidx in range(liver_imgs_test.shape[0]):
        orig_img = np.copy(liver_imgs_test[sliceidx, :, :])
        liver_imgs_test[sliceidx, :, :] = gaussian_filter(np.float32(orig_img), sigma=2)
    liver_imgs_test = liver_imgs_test[..., np.newaxis]
    liver_imgs_test -= liver_mean
    liver_imgs_test /= liver_std

    liver_mask_test = livermodel.predict(liver_imgs_test, verbose=1)

    selectedmaskpx = 0
    selectedidx = 0

    for sliceidx in range(liver_mask_test.shape[0]):
        maskpx = np.count_nonzero(liver_mask_test[sliceidx, :, :, 0] == 1)
        if selectedmaskpx < maskpx:
            selectedidx = sliceidx
            selectedmaskpx = maskpx

        maxliverimg = liver_img3D[sliceidx, :, :]
        maxliverimg = cv2.resize(maxliverimg, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        offset = 100
        upperl = 255 - offset
        lowerl = 0 - offset
        maxliverimg[maxliverimg > upperl] = upperl
        maxliverimg[maxliverimg < lowerl] = lowerl
        maxliverimg += offset
        maxliverimg *= 1
        maxlivermsk = liver_mask_test[sliceidx, :, :, 0].astype('uint8')
        selectedimg = maxliverimg * maxlivermsk
        selectedimg.astype('uint8')
        imgfname = os.path.join(save_path, caseid +'_'+str(sliceidx)+ '.png')
        print(imgfname)

        try:
            segmentation = models.Segmentation(caseID=caseid, segID=sliceidx, seg=imgfname, is_selected=False)
            segmentation.save()
        except:
            print('未保存成功')
        imgdata = im.fromarray(selectedimg)
        imgdata = imgdata.convert('RGB')
        imgdata.save(imgfname)

    try:
        models.Segmentation.objects.filter(caseID=caseid, segID=selectedidx).update(is_selected = True)
    except:
        print('未修改成功')

# checkpoint
CHECKPOINT_PATH = sys.path[0]+'/model/pathology-cnn-clf.ckpt'

# arguement setting
NUM_CLASSES = 8

# dict of classes name
className = {0: 'cirrhosis_only',
             1: 'cirrhosis_viral_hepatitis',
             2: 'hcc_cirrhosis',
             3: 'hcc_only',
             4: 'hcc_viral_hepatitis',
             5: 'hcc_viral_hepatitis_cirrhosis',
             6: 'normal_liver',
             7: 'viral_hepatitis_only'}

# Create your views here.
def segmentation(request):
    path = request.GET.get('PATH')
    caseid = request.GET.get('ID')

    srcfolder = r'.\demo01\static\images'
    if not os.path.exists(srcfolder):
        os.mkdir(srcfolder)

    casefolder = os.path.join(srcfolder,caseid)
    if not os.path.exists(casefolder):
        os.mkdir(casefolder)

    slice_selection(casefolder_path=path, save_path=casefolder, caseid=caseid)

    return HttpResponse("OK")

def classification(request):
    caseid = request.GET.get('ID')
    print("ID",int(caseid))
    selected_seg = models.Segmentation.objects.filter(caseID=caseid,is_selected=True)
    img_path = list(selected_seg.values())[0].get('seg')


    img_arr = plt.imread(img_path)
    img_arr_new = np.float32(cv2.resize(img_arr, (224, 224)))
    img_arr_new = img_arr_new.transpose(2, 0, 1)  # convert HWC to CHW
    img_arr_new = img_arr_new[np.newaxis, :, :, :, ]
    print("img_arr_new:", np.array(img_arr_new).shape)
    X = Tensor(img_arr_new, mstype.float32)
    print("X.shape",X.shape)

    net = resnet.resnet50(NUM_CLASSES)

    param_dict = load_checkpoint(CHECKPOINT_PATH)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net)

    Y = model.predict(X)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    print("Y:", Y)
    print("Y:", type(Y))
    # print("softmax:", softmax(Y[0]))

    tempList = Y[0].asnumpy()
    resultList = tempList.tolist()
    print(softmax(resultList))
    print([x*100 for x in softmax(resultList)])

    idx = np.argmax(tempList)
    result = className[idx]
    print('The Case is: %s' % ( result))
    print('\n')
    return HttpResponse(result)

def __pca_process(m):
    #instanitiate pca
    pca = PCA(n_components=2)
    #implement pca over the concated array
    principalComponents = pca.fit_transform(m)
    variance_account = pca.explained_variance_
    return principalComponents,variance_account

def Cluster(request):
    caseid = request.GET.get('ID')
    print("ID", int(caseid))
    selected_seg = models.Segmentation.objects.filter(caseID=caseid, is_selected=True)
    img_path = list(selected_seg.values())[0].get('seg')

    param_dict = load_checkpoint(CHECKPOINT_PATH)
    print("param_dict", len(param_dict))
    sub_param_dic = {}
    length = len(param_dict)
    parameters = list(param_dict.keys())
    for i in range(length - 2):
        sub_param_dic[parameters[i]] = param_dict[parameters[i]]

    net = resnet.resnet50_(NUM_CLASSES)
    load_param_into_net(net, sub_param_dic)
    net.set_train(False)
    model = Model(net)

    img_arr = plt.imread(img_path)
    img_arr_new = np.float32(cv2.resize(img_arr, (224, 224)))
    img_arr_new = img_arr_new.transpose(2, 0, 1)  # convert HWC to CHW
    img_arr_new = img_arr_new[np.newaxis, :, :, :, ]
    X = Tensor(img_arr_new, mstype.float32)
    Y = model.predict(X)
    numpy_Y = Y.asnumpy()
    print("type(Y):",type(numpy_Y))
    # cluster
    className = {0: 'Cirrhosis only',
                 1: 'Cirrhosis & Viral Hepatitis',
                 2: 'HCC & Cirrhosis',
                 3: 'HCC only',
                 4: 'HCC & Viral Hepatitis',
                 5: 'HCC & Viral Hepatitis & Cirrhosis',
                 6: 'Normal Liver',
                 7: 'Viral Hepatitis only',
                 -1: 'Query Case'}
    DATA_PATH = sys.path[0] + '/model/dataframe.csv'
    trainframe = pd.read_csv(DATA_PATH, index_col=0)
    x_train = trainframe.iloc[:, :50].values
    y_train = trainframe.iloc[:, 50].values
    pcacom, pca_var_explained = __pca_process(np.concatenate((x_train, numpy_Y), axis=0))

    principalDf = pd.DataFrame(data=pcacom
                               , columns=['principal component 1', 'principal component 2'])
    y = np.append(y_train, -1)
    principalDf['Legend'] = pd.Series(y).map(className)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Two principal neurons', fontsize=20)
    targets = list(className.values())
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:red']
    for target, color in zip(targets, colors):
        if target == 'Query Case':
            ax.scatter(principalDf.loc[397, 'principal component 1']
                       , principalDf.loc[397, 'principal component 2']
                       , c=color
                       , s=150)
        else:
            indicesToKeep = principalDf['Legend'] == target
            ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                       , principalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=20)
    ax.legend(targets, loc=2)
    ax.grid()
    clusterfolder = r'.\clusterImg'
    if not os.path.exists(clusterfolder):
        os.mkdir(clusterfolder)



    imgfname = os.path.join(clusterfolder, caseid + '.png')
    print("Cluster Img Path:",imgfname)

    try:
        plt.savefig(imgfname)
        clusterImg = models.ClusterImg(caseID=caseid, imgPath=imgfname)
        clusterImg.save()
    except:
        print('未保存成功')

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(pcacom[:398, :], y_train)
    _, index = clf.kneighbors(pcacom[397, :].reshape(1, -1))
    index = index.flatten().tolist()
    Case_name = [trainframe.index[x] for x in index]
    Case_ID = [CaseID[4:7] for CaseID in Case_name]
    # neighbors = [className[y_train[x]] for x in index]

    return HttpResponse(Case_ID)

def tocaseid(l):
    # l = "Case%s.png" % l
    return l

def remove_sublist(target, rlist):
    for num in rlist:
        if num in target:
            target.remove(num)
    return target

class pca_knn():

    def __init__(self,querycase) -> None:
        '''
        Only the number of querycase is needed
        '''
        #load the training set

        self.className = {0: 'Cirrhosis only',
             1: 'Cirrhosis & Viral Hepatitis',
             2: 'HCC & Cirrhosis',
             3: 'HCC only',
             4: 'HCC & Viral Hepatitis',
             5: 'HCC & Viral Hepatitis & Cirrhosis',
             6: 'Normal Liver',
             7: 'Viral Hepatitis only',
             -1: 'Query Case'}
        DATA_PATH = sys.path[0] + '/model/dataframe1.csv'
        # trainframe = pd.read_csv(DATA_PATH, index_col=0)
        df = pd.read_csv(DATA_PATH,index_col = 0)
        self.querycase = tocaseid(querycase)
        print('self.querycase:',self.querycase)
        # split the data set to query set and train set
        queryframe = df.loc[self.querycase]

        print("queryframe:",queryframe.iloc[:50].values.reshape(1,-1))
        print("queryframe.type:",type(queryframe.iloc[:50].values.reshape(1,-1)))
        trainframe = df.drop(self.querycase)
        self.trainframe = df.drop(self.querycase)
        self.x_query = queryframe.iloc[:50].values.reshape(1,-1)
        self.y_query = queryframe.iloc[50]
        self.x_train = trainframe.iloc[:,:50].values
        self.y_train = trainframe.iloc[:,50].values
        self.querylabel = self.className[self.y_query]
        self.pcacom, self.pca_var_explained = self.__pca_process(np.concatenate((self.x_train,self.x_query),axis=0))

    def __pca_process(self,m):
        #instanitiate pca
        pca = PCA(n_components=2)
        #implement pca over the concated array
        principalComponents = pca.fit_transform(m)
        variance_account = pca.explained_variance_
        return principalComponents,variance_account


    def pca_plot(self):
        '''
        Plot the clustering and show the query case on the plot
        '''
        principalDf = pd.DataFrame(data = self.pcacom
             , columns = ['principal component 1', 'principal component 2'])
        y = np.append(self.y_train,-1)
        principalDf['Legend'] = pd.Series(y).map(self.className)
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('Two principal neurons', fontsize = 20)
        targets = list(self.className.values())
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray','tab:red']
        for target, color in zip(targets,colors):
            if target == 'Query Case':
                ax.scatter(principalDf.loc[397, 'principal component 1']
                    , principalDf.loc[397, 'principal component 2']
                    , c = color
                    , s = 150)
            else:
                indicesToKeep = principalDf['Legend'] == target
                ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                        , principalDf.loc[indicesToKeep, 'principal component 2']
                        , c = color
                        , s = 20)
        ax.legend(targets,loc=2)
        ax.grid()


    def knn_process(self):
        '''
        return: tuple(list[String], list[String])
        the 7 nearest cases' label based on the query case and the corresponding id
        '''
        clf = KNeighborsClassifier(n_neighbors=7)
        clf.fit(self.pcacom[:397,:],self.y_train)
        _, index = clf.kneighbors(self.pcacom[397,:].reshape(1,-1))
        index = index.flatten().tolist()
        Case_name = [self.trainframe.index[x] for x in index]
        neighbors = [self.className[self.y_train[x]] for x in index]
        return neighbors,Case_name



def Report(request):
    caseid = request.GET.get('ID')
    print("ID:", int(caseid))
    query_case = models.Segmentation.objects.filter(caseID=caseid)
    query_segmentations = list(query_case.values())
    segImages = [query_segmentations[i].get("seg")[22:].replace("\\","/") for i in range(len(query_segmentations))]


    selected_seg = models.Segmentation.objects.filter(caseID=caseid,is_selected=True)
    selected_seg_ID = list(selected_seg.values())[0].get('segID')
    target_segmentation_path = [segImages[i] for i in range(len(segImages))]

    for i in range(len(segImages)):
        if(i<selected_seg_ID):
            target_segmentation_path[len(query_segmentations)-selected_seg_ID+i] = segImages[i]
        else:
            target_segmentation_path[i-selected_seg_ID] = segImages[i]

    # classify
    img_path = list(selected_seg.values())[0].get('seg')


    img_arr = plt.imread(img_path)
    img_arr_new = np.float32(cv2.resize(img_arr, (224, 224)))
    img_arr_new = img_arr_new.transpose(2, 0, 1)  # convert HWC to CHW
    img_arr_new = img_arr_new[np.newaxis, :, :, :, ]
    X = Tensor(img_arr_new, mstype.float32)

    net = resnet.resnet50(NUM_CLASSES)

    param_dict = load_checkpoint(CHECKPOINT_PATH)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net)

    Y = model.predict(X)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    tempList = Y[0].asnumpy()
    resultList = tempList.tolist()
    softmaxlist =[softmax(resultList)[i]*100 for i in range(len(softmax(resultList)))]
    targetDict = {}
    for i in range(len(softmaxlist)):
        if softmaxlist[i]>2:
            targetDict[className[i]] = np.around(softmaxlist[i],2)
    idx = np.argmax(tempList)



    # cluster
    cluster_className = {0: 'Cirrhosis only',
                 1: 'Cirrhosis & Viral Hepatitis',
                 2: 'HCC & Cirrhosis',
                 3: 'HCC only',
                 4: 'HCC & Viral Hepatitis',
                 5: 'HCC & Viral Hepatitis & Cirrhosis',
                 6: 'Normal Liver',
                 7: 'Viral Hepatitis only',
                 -1: 'Query Case'}
    cluster_param_dict = load_checkpoint(CHECKPOINT_PATH)
    cluster_sub_param_dic = {}
    length = len(cluster_param_dict)
    parameters = list(cluster_param_dict.keys())
    for i in range(length - 2):
        cluster_sub_param_dic[parameters[i]] = cluster_param_dict[parameters[i]]

    cluster_net = resnet.resnet50_(NUM_CLASSES)
    load_param_into_net(cluster_net, cluster_sub_param_dic)
    cluster_net.set_train(False)
    cluster_model = Model(cluster_net)

    cluster_img_arr = plt.imread(img_path)
    cluster_img_arr_new = np.float32(cv2.resize(cluster_img_arr, (224, 224)))
    cluster_img_arr_new = cluster_img_arr_new.transpose(2, 0, 1)  # convert HWC to CHW
    cluster_img_arr_new = cluster_img_arr_new[np.newaxis, :, :, :, ]

    cluster_X = Tensor(cluster_img_arr_new, mstype.float32)
    numpy_Y = cluster_model.predict(cluster_X).asnumpy()
    print("numpy_Y:",numpy_Y)

    latter_layer_output = np.hstack(numpy_Y)
    extract_output = latter_layer_output.T
    extract_output = np.asarray(extract_output, 'int64')
    numpy_idx = np.array([[idx]])
    extract_output = StandardScaler().fit_transform(extract_output.reshape(1,50))
    result_Y = np.concatenate((numpy_Y, numpy_idx), axis=1)

    DATA_PATH = sys.path[0] + '/model/New.csv'
    trainframe = pd.read_csv(DATA_PATH, index_col=0)
    index_list = trainframe.index.values.tolist()
    current_case_In_csv = False
    for index in index_list:
        if int(caseid)==index:
            current_case_In_csv = True
            break

    if not current_case_In_csv:
        print("Cuurent case not in .CSV")
        index_list.append(caseid)
        Test_PATH = sys.path[0] + '/model/New.csv'

        pandas_colume = []

        for i in range(50):
            pandas_colume.append(i)

        pandas_colume.append("label")
        print("trainframe.size:",trainframe.size)
        print("numpy_Y.size:",numpy_Y.size)

        new_numpyfile = np.concatenate((trainframe,result_Y), axis=0)
        pd.DataFrame(data=new_numpyfile, columns=pandas_colume,index=index_list).to_csv(Test_PATH)
    else:
        print("current Case in .CSV")
    count_csv = int(trainframe.shape[0])

    x_train = trainframe.iloc[:, :50].values
    y_train = trainframe.iloc[:, 50].values
    print("x_train.size:",x_train.size)
    print("extract_output.size:",extract_output.size)


    pcacom, pca_var_explained = __pca_process(np.concatenate((x_train, numpy_Y), axis=0))

    principalDf = pd.DataFrame(data=pcacom, columns=['principal component 1', 'principal component 2'])
    cluster_y = np.append(y_train, -1)
    principalDf['Legend'] = pd.Series(cluster_y).map(cluster_className)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Two principal neurons', fontsize=20)
    targets = list(cluster_className.values())
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:red']
    for target, color in zip(targets, colors):
        if target == 'Query Case':
            ax.scatter(principalDf.loc[count_csv, 'principal component 1']
                       , principalDf.loc[count_csv, 'principal component 2']
                       , c=color
                       , s=150)
        else:
            indicesToKeep = principalDf['Legend'] == target
            ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                       , principalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=20)
    ax.legend(targets, loc=2)
    ax.grid()

    clusterfolder = r'.\demo01\static\clusterImg'
    if not os.path.exists(clusterfolder):
        os.mkdir(clusterfolder)

    imgfname = os.path.join(clusterfolder, caseid + '.png')

    try:
        plt.savefig(imgfname)
        clusterImg = models.ClusterImg(caseID=caseid, imgPath=imgfname)
        clusterImg.save()
    except:
        print('未保存成功')

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(pcacom[:count_csv, :], y_train)
    _, index = clf.kneighbors(pcacom[count_csv, :].reshape(1, -1))
    index = index.flatten().tolist()
    Case_name = [trainframe.index[x] for x in index]

    met = -1
    if(int(caseid)>0 and int(caseid)<=180):
        Transfer_DATA_PATH = sys.path[0] + '/model/Transfer.csv'
        df = pd.read_csv(Transfer_DATA_PATH, index_col=0)
        Transfer_queryframe = df.loc[int(caseid)]
        met = Transfer_queryframe['met']

    return render(request, "viewer_old.html",{"ID":caseid,"target_segmentation_path":target_segmentation_path,"targetDict":targetDict,"clusterImg": imgfname[26:],"knn_list":Case_name,"met":met})


def case(request):

    ID = request.GET.get('ID')
    res = models.Category.objects.filter(Case_No = ID)
    category =  list(res.values())[0]
    my_Internal_case_code = list(res.values('Internal_case_code'))[0].get('Internal_case_code')

    category.pop('Internal_ID')
    category.pop('Internal_case_code')


    if(my_Internal_case_code!=-1 and my_Internal_case_code != None):
        response_detail = list(models.Detail.objects.filter(Internal_case_code = my_Internal_case_code).values())[0]
        response_detail.pop('Internal_case_code')
        response_detail.pop('Internal_ID')
        response_detail.pop('Code')
        response_detail.pop('Gender')

        return render(request,'excel.html',{"category":category,"detail":response_detail,"ID":ID})
    else:
        return render(request,'excel.html',{"category":category,"ID":ID})


