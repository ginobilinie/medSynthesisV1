import numpy as np
import os
import SimpleITK as sitk
import h5py
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init
from torch.autograd import Variable
from scipy import signal
from scipy import ndimage
import ast
import argparse
import copy
import gauss

#Dong add keys here
def Generator_2D_slices(path_patients,batchsize,inputKey='dataMR',outputKey='dataCT'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient),'r')
            #dataMRptr=f['dataMR']
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            
            #dataCTptr=f['dataCT']
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)

            #print 'mr shape h5 ',dataMR.shape#B,H,W,C
            #print 'ct shape h5 ',dataCT.shape#B,H,W
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=3)#B,H,W,C
            y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]
            
            print 'y shape ', y.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])


# Dong add keys here
# We extract items from one whole Epoch: traverse the data thoroughly
def Generator_2D_slices_OneEpoch(path_patients, batchsize, inputKey='dataMR', outputKey='dataCT'):
    # path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)  # every file  is a hdf5 patient
    # while True:
    #
    for idx, namepatient in enumerate(patients):
        print namepatient
        f = h5py.File(os.path.join(path_patients, namepatient), 'r')
        # dataMRptr=f['dataMR']
        dataMRptr = f[inputKey]
        dataMR = dataMRptr.value

        # dataCTptr=f['dataCT']
        dataCTptr = f[outputKey]
        dataCT = dataCTptr.value

        dataMR = np.squeeze(dataMR)
        dataCT = np.squeeze(dataCT)

        # print 'mr shape h5 ',dataMR.shape#B,H,W,C
        # print 'ct shape h5 ',dataCT.shape#B,H,W

        shapedata = dataMR.shape
        # Shuffle data
        idx_rnd = np.random.choice(shapedata[0], shapedata[0], replace=False)
        dataMR = dataMR[idx_rnd, ...]
        dataCT = dataCT[idx_rnd, ...]
        modulo = np.mod(shapedata[0], batchsize)
        ################## always the number of samples will be a multiple of batchsz##########################3
        if modulo != 0:
            to_add = batchsize - modulo
            inds_toadd = np.random.randint(0, dataMR.shape[0], to_add)
            X = np.zeros((dataMR.shape[0] + to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))  # dataMR
            X[:dataMR.shape[0], ...] = dataMR
            X[dataMR.shape[0]:, ...] = dataMR[inds_toadd]

            y = np.zeros((dataCT.shape[0] + to_add, dataCT.shape[1], dataCT.shape[2]))  # dataCT
            y[:dataCT.shape[0], ...] = dataCT
            y[dataCT.shape[0]:, ...] = dataCT[inds_toadd]

        else:
            X = np.copy(dataMR)
            y = np.copy(dataCT)

        # X = np.expand_dims(X, axis=3)
        X = X.astype(np.float32)
        y = np.expand_dims(y, axis=3)  # B,H,W,C
        y = y.astype(np.float32)
        # y[np.where(y==5)]=0

        # shuffle the data, by dong
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds, ...]
        y = y[inds, ...]

        print 'y shape ', y.shape
        for i_batch in xrange(int(X.shape[0] / batchsize)):
            yield (X[i_batch * batchsize:(i_batch + 1) * batchsize, ...],
                   y[i_batch * batchsize:(i_batch + 1) * batchsize, ...])



#Dong add keys here
def Generator_2D_slicesV1(path_patients,batchsize,inputKey='dataMR',segKey='dataCT', contourKey='dataContour'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient),'r')
            #dataMRptr=f['dataMR']
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            
            #dataCTptr=f['dataCT']
            dataCTptr=f[segKey]
            dataCT=dataCTptr.value

            dataContourptr=f[contourKey]
            dataContour=dataContourptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)
            dataContour=np.squeeze(dataContour)

            #print 'mr shape h5 ',dataMR.shape#B,H,W,C
            print 'ct shape h5 ',dataCT.shape#B,H,W
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            dataContour=dataContour[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2],dataCT.shape[3]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
                y1=np.zeros((dataContour.shape[0]+to_add,dataContour.shape[1],dataContour.shape[2]))#dataCT
                y1[:dataContour.shape[0],...]=dataContour
                y1[dataContour.shape[0]:,...]=dataContour[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)
                y1 = np.copy(dataContour)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
#             y=np.expand_dims(y, axis=3)#B,H,W,C
#             y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]
            y1=y1[inds,...]
            
            print 'y shape ', y.shape                   
            print 'X shape ', X.shape                   
            print 'y1 shape ', y1.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...],y1[i_batch*batchsize:(i_batch+1)*batchsize,...])


# Dong add keys here
# We extract items from one whole Epoch: traverse the data thoroughly
def Generator_2D_slicesV1_OneEpoch(path_patients, batchsize, inputKey='dataMR', segKey='dataCT', contourKey='dataContour'):
    # path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)  # every file  is a hdf5 patient
    # # while True:
    #
    for idx, namepatient in enumerate(patients):
        print namepatient
        f = h5py.File(os.path.join(path_patients, namepatient), 'r')
        # dataMRptr=f['dataMR']
        dataMRptr = f[inputKey]
        dataMR = dataMRptr.value

        # dataCTptr=f['dataCT']
        dataCTptr = f[segKey]
        dataCT = dataCTptr.value

        dataContourptr = f[contourKey]
        dataContour = dataContourptr.value

        dataMR = np.squeeze(dataMR)
        dataCT = np.squeeze(dataCT)
        dataContour = np.squeeze(dataContour)

        # print 'mr shape h5 ',dataMR.shape#B,H,W,C
        print 'ct shape h5 ', dataCT.shape  # B,H,W

        shapedata = dataMR.shape
        # Shuffle data
        idx_rnd = np.random.choice(shapedata[0], shapedata[0], replace=False)
        dataMR = dataMR[idx_rnd, ...]
        dataCT = dataCT[idx_rnd, ...]
        dataContour = dataContour[idx_rnd, ...]
        modulo = np.mod(shapedata[0], batchsize)
        ################## always the number of samples will be a multiple of batchsz##########################3
        if modulo != 0:
            to_add = batchsize - modulo
            inds_toadd = np.random.randint(0, dataMR.shape[0], to_add)
            X = np.zeros((dataMR.shape[0] + to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))  # dataMR
            X[:dataMR.shape[0], ...] = dataMR
            X[dataMR.shape[0]:, ...] = dataMR[inds_toadd]

            y = np.zeros((dataCT.shape[0] + to_add, dataCT.shape[1], dataCT.shape[2], dataCT.shape[3]))  # dataCT
            y[:dataCT.shape[0], ...] = dataCT
            y[dataCT.shape[0]:, ...] = dataCT[inds_toadd]

            y1 = np.zeros((dataContour.shape[0] + to_add, dataContour.shape[1], dataContour.shape[2]))  # dataCT
            y1[:dataContour.shape[0], ...] = dataContour
            y1[dataContour.shape[0]:, ...] = dataContour[inds_toadd]

        else:
            X = np.copy(dataMR)
            y = np.copy(dataCT)
            y1 = np.copy(dataContour)

        # X = np.expand_dims(X, axis=3)
        X = X.astype(np.float32)
        #             y=np.expand_dims(y, axis=3)#B,H,W,C
        #             y=y.astype(np.float32)
        # y[np.where(y==5)]=0

        # shuffle the data, by dong
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds, ...]
        y = y[inds, ...]
        y1 = y1[inds, ...]

        print 'y shape ', y.shape
        print 'X shape ', X.shape
        print 'y1 shape ', y1.shape
        for i_batch in xrange(int(X.shape[0] / batchsize)):
            yield (X[i_batch * batchsize:(i_batch + 1) * batchsize, ...],
                   y[i_batch * batchsize:(i_batch + 1) * batchsize, ...],
                   y1[i_batch * batchsize:(i_batch + 1) * batchsize, ...])


#Dong add a variable of keys here
'''
Input:
    path_patients: h5 data path
    batchsize: the batchsize we extract patches at a time
    keys: a variable number of parameters
Output:
    The list of corresponding values indexed by the keys
'''
def Generator_2D_slices_variousKeys(path_patients,batchsize, *keys):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    numOfKeys = len(keys)
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f = h5py.File(os.path.join(path_patients,namepatient))
            
            data0 = f[keys[0]].value
            shapedata=data0.shape
#             idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            assert len(shapedata) == 4, 'data should have shape like: NxCxHxW'
            keyvalue = np.zeros((shapedata[0],shapedata[1],shapedata[2],shapedata[3],shapedata[4],numOfKeys))
            
            for keyInd in range(0,numOfKeys):
                #dataMRptr=f['dataMR']
                key = keys[keyInd]
                keyvalue[:,:,:,:,keyInd] = f[key].value
            
#                 dataMR=np.squeeze(dataMR)
#                 dataCT=np.squeeze(dataCT)
    
                #print 'mr shape h5 ',dataMR.shape#B,H,W,C
                #print 'ct shape h5 ',dataCT.shape#B,H,W
                
                
                #Shuffle data
            idx_rnd = np.random.choice(shapedata[0], shapedata[0], replace=False)
            keyvalue = keyvalue[idx_rnd,...]

            
            modulo = np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0: #we consider the remaining parts (e.g., 10008%8 = 2)
                to_add = batchsize-modulo
                inds_toadd = np.random.randint(0,keyvalue.shape[0],to_add)
                X = np.zeros((keyvalue.shape[0]+to_add,keyvalue.shape[1],keyvalue.shape[2],keyvalue.shape[3], keyvalue.shape[4]))#keyvalue
                X[:keyvalue.shape[0],...] = keyvalue
                X[keyvalue.shape[0]:,...] = keyvalue[inds_toadd]                
                
#                 y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
#                 y[:dataCT.shape[0],...]=dataCT
#                 y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X = np.copy(keyvalue)
#                 X=np.copy(dataMR)                
#                 y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
#             y=np.expand_dims(y, axis=3)#B,H,W,C
#             y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
#             y=y[inds,...]
            
            print 'X shape ', X.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
#                 yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...,keyInd],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])
                yield ([ X[i_batch*batchsize:(i_batch+1)*batchsize,...,keyInd] for keyInd in range(0,numOfKeys)])




def Generator_3D_patches(path_patients,batchsize, inputKey='dataMR',outputKey='dataCT'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            #dataMR=np.squeeze(dataMR)
            
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value
            #dataCT=np.squeeze(dataCT)

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)
            print 'mr shape h5 ',dataMR.shape

            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0, dataMR.shape[0], to_add)
                X=np.zeros((dataMR.shape[0]+to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add, dataCT.shape[1], dataCT.shape[2], dataCT.shape[3]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            X = np.expand_dims(X, axis=4)     
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=4)
            y=y.astype(np.float32)
            
            print 'y shape ', y.shape
            print 'X shape ', X.shape
                             
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])
                
'''
    custom weights initialization called on netG and netD
    I think this can only goes into the 1st space, instead of recursive initialization
 '''
def weights_init(m):
    xavier=torch.nn.init.xavier_uniform
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
        #print m.weight.data
        #print m.bias.data
        xavier(m.weight.data)
#         print 'come xavier'
        #xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

'''
    this function is used to compute the dice ratio
input:
    im1: gt
    im2 pred
    tid: the id for consideration
output:
    dcs
'''
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


'''
    this function is used to compute psnr
input:
    ct_generated and ct_GT
output:
    psnr
'''
def psnr(ct_generated,ct_GT):
    print ct_generated.shape
    print ct_GT.shape

    mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
    print 'mse ',mse
    max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
    print 'max_I ',max_I
    return 20.0*np.log10(max_I/mse)

'''
  this function is used to compute ssim
  input:
       ct_generated and ct_gt
  output:
       ssim
'''
def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
    

'''
    for finetune or sth else to transfer the weights from other models
'''
def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)



'''
    Evaluate one patch using the latest pytorch model
input:
    patch_MR: a np array of shape [H,W,nchans]
output:    
    patch_CT_pred: segmentation maps for the corresponding input patch 
'''
def evaluate(patch_MR, netG, modelPath):
    
    
        patch_MR = torch.from_numpy(patch_MR)

        patch_MR = patch_MR.unsqueeze(0)
#         patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
        patch_MR = Variable(patch_MR).float().cuda()
#         netG = ResSegNet() #here is your network
#         netG.load_state_dict(torch.load(modelPath))
        netG.cuda()
        netG.eval()
#         print type(patch_MR)
        res = netG(patch_MR)
        
#         print res.size(),res.squeeze(0).size()
        if isinstance(res, tuple):
            res = res[0]
        _, tmp = res.squeeze(0).max(0)
        patchOut = tmp.data.cpu().numpy().squeeze()

        #imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
        #imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
        #print 'mean of layer  ',np.mean(MR16_eval)
        #print 'min ct estimated ',np.min(patch_CT_pred)
        #print 'max ct estimated ',np.max(patch_CT_pred)
        #print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
        return patchOut

'''
    Evaluate one patch using the latest pytorch model
input:
    patch_MR: a np array of shape [H,W,nchans]
output:    
    patch_CT_pred: segmentation maps for the corresponding input patch 
'''
def evaluate_reg(patch_MR, netG, modelPath):
    patch_MR = torch.from_numpy(patch_MR)

    patch_MR = patch_MR.unsqueeze(0)
    #         patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
    patch_MR = Variable(patch_MR).float().cuda()
    #         netG = ResSegNet() #here is your network
    #         netG.load_state_dict(torch.load(modelPath))
    netG.cuda()
    netG.eval()
    #         print type(patch_MR)
    res = netG(patch_MR)

    #         print res.size(),res.squeeze(0).size()
    if isinstance(res, tuple):
        res = res[0]
    # _, tmp = res.squeeze(0).max(0)
    tmp = res
    patchOut = tmp.data.cpu().numpy().squeeze()

    # imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
    # imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
    # print 'mean of layer  ',np.mean(MR16_eval)
    # print 'min ct estimated ',np.min(patch_CT_pred)
    # print 'max ct estimated ',np.max(patch_CT_pred)
    # print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
    return patchOut

'''
    Evaluate one patch with long-term skip connection using the latest pytorch model
input:
    patch_MR: a np array of shape [H,W,nchans]
output:    
    patch_CT_pred: prediction maps for the corresponding input patch 
'''


def evaluate_res(patch_MR, netG, modelPath, nd=2):
    patch_MR = torch.from_numpy(patch_MR)

    patch_MR = patch_MR.unsqueeze(0)

    shape = patch_MR.shape
    chn = shape[1]
    #print 'channel is ',chn
    if nd!=2:
        patch_MR = patch_MR.unsqueeze(0)
        res_MR = patch_MR
    else:
        mid_slice = chn//2
        res_MR = patch_MR[:,mid_slice,...]

    #         patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
    patch_MR = Variable(patch_MR.float().cuda())
    res_MR = Variable(res_MR.float().cuda())


    #         netG = ResSegNet() #here is your network
    #         netG.load_state_dict(torch.load(modelPath))
    netG.cuda()
    netG.eval()
    #         print type(patch_MR)
    res = netG(patch_MR,res_MR)

    #         print res.size(),res.squeeze(0).size()
    if isinstance(res, tuple):
        res = res[0]
    # _, tmp = res.squeeze(0).max(0)
    tmp = res
    patchOut = tmp.data.cpu().numpy().squeeze()

    # imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
    # imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
    # print 'mean of layer  ',np.mean(MR16_eval)
    # print 'min ct estimated ',np.min(patch_CT_pred)
    # print 'max ct estimated ',np.max(patch_CT_pred)
    # print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
    return patchOut

'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use averaging at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def testOneSubject_aver_res(MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath, nd=2):

    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz

    eps = 1e-5
    [row,col,leng] = matFA.shape
    margin1 = int((dFA[0]-dSeg[0])/2)
    margin2 = int((dFA[1]-dSeg[1])/2)
    margin3 = int((dFA[2]-dSeg[2])/2)
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

    matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
    matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension

    matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
    matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension

    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]


    matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    #fid=open('trainxxx_list.txt','a');
    print 'last i ',row-dSeg[0]
    for i in range(0,row-dSeg[0]+1,step[0]):
#         print 'i ',i
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                #print 'volFA shape is ',volFA.shape
                #mynet.blobs['dataMR'].data[0,0,...]=volFA
                #mynet.forward()
                #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                #print 'max: ',np.amax(volFA),'min: ',np.amin(volFA)
                temppremat = evaluate_res(volFA, netG, modelPath, nd=nd)
                #print 'max: ',np.amax(temppremat),'min: ',np.amin(temppremat)
#                 print 'temppremat shape 1: ',temppremat.shape
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                #print 'patchout shape ',temppremat.shape
                #temppremat=volSeg
#                 print 'temppremat shape 2: ',temppremat.shape
                matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
    matOut = matOut/used
    return matOut

'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use averaging at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def testOneSubject_aver_res_multiModal(FA_image, MR_image, CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath):

    matFA = FA_image
    matMR = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz

    eps = 1e-5
    [row,col,leng] = matFA.shape
    margin1 = int((dFA[0]-dSeg[0])/2)
    margin2 = int((dFA[1]-dSeg[1])/2)
    margin3 = int((dFA[2]-dSeg[2])/2)
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

    matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
    matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension

    matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
    matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension

    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]

    matMROut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matMROut shape is ',matMROut.shape
    matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matMR

    matMROut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matMR[0:marginD[0],:,:] #we'd better flip it along the first dimension
    matMROut[row+marginD[0]:matMROut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matMR[row-marginD[0]:matMR.shape[0],:,:] #we'd better flip it along the 1st dimension

    matMROut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matMR[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
    matMROut[marginD[0]:row+marginD[0],col+marginD[1]:matMROut.shape[1],marginD[2]:leng+marginD[2]] = matMR[:,col-marginD[1]:matMR.shape[1],:] #we'd better to flip it along the 2nd dimension

    matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matMR[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
    matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matMROut.shape[2]] = matMR[:,:,leng-marginD[2]:matMR.shape[2]]


    matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    #fid=open('trainxxx_list.txt','a');
    print 'last i ',row-dSeg[0]
    for i in range(0,row-dSeg[0]+1,step[0]):
#         print 'i ',i
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                volMR = matMROut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                volMat = np.concatenate((volFA, volMR), 0)
                #print 'volFA shape is ',volFA.shape
                #mynet.blobs['dataMR'].data[0,0,...]=volFA
                #mynet.forward()
                #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                temppremat = evaluate_res(volMat, netG, modelPath)
#                 print 'temppremat shape 1: ',temppremat.shape
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                #print 'patchout shape ',temppremat.shape
                #temppremat=volSeg
#                 print 'temppremat shape 2: ',temppremat.shape
                matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
    matOut = matOut/used
    return matOut



'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use averaging at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation/regression map
'''
def testOneSubject_aver(MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath, type='reg'):

    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    eps = 1e-5
    [row,col,leng] = matFA.shape
    margin1 = int((dFA[0]-dSeg[0])/2)
    margin2 = int((dFA[1]-dSeg[1])/2)
    margin3 = int((dFA[2]-dSeg[2])/2)
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA
    
    matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
    matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
    
    matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
    matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
    
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    
    
    matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    #fid=open('trainxxx_list.txt','a');
    print 'last i ',row-dSeg[0]
    for i in range(0,row-dSeg[0]+1,step[0]):
#         print 'i ',i
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                #print 'volFA shape is ',volFA.shape
                #mynet.blobs['dataMR'].data[0,0,...]=volFA
                #mynet.forward()
                #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                if type == 'reg':
                    temppremat = evaluate_reg(volFA, netG, modelPath)
                else:
                    temppremat = evaluate(volFA, netG, modelPath)
                # temppremat = evaluate(volFA, netG, modelPath)
#                 print 'temppremat shape 1: ',temppremat.shape
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                #print 'patchout shape ',temppremat.shape
                #temppremat=volSeg
#                 print 'temppremat shape 2: ',temppremat.shape
                matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
    matOut = matOut/used
    return matOut


'''
    Receives mulit-modal MR image and returns an segmentation label maps with the same size
    We use averaging at the overlapping regions
input:
    FA_image: a first modality, the raw input data (after preprocessing)
    MR_image: a second modality, the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation/regression map
'''
def testOneSubject_aver_MultiModal(FA_image, MR_image, CT_GT, MR_patch_sz, CT_patch_sz, step, netG, modelPath, type='reg'):
    matFA = FA_image
    matMR = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz

    eps = 1e-5
    [row, col, leng] = matFA.shape
    margin1 = int((dFA[0] - dSeg[0]) / 2)
    margin2 = int((dFA[1] - dSeg[1]) / 2)
    margin3 = int((dFA[2] - dSeg[2]) / 2)
    cubicCnt = 0
    marginD = [margin1, margin2, margin3]
    print 'matFA shape is ', matFA.shape
    matFAOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]])
    print 'matFAOut shape is ', matFAOut.shape
    matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA

    print 'matMR shape is ', matMR.shape
    matMROut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]])
    print 'matMROut shape is ', matMROut.shape
    matMROut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMR

    ## matFA
    # we'd better flip it along the first dimension
    matFAOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[0:marginD[0], :,:]
    # we'd better flip it along the 1st dimension
    matFAOut[row + marginD[0]:matFAOut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[row - marginD[0]:matFA.shape[0], :,:]

    # we'd better flip it along the 2nd dimension
    matFAOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matFA[:, 0:marginD[1],:]
    # we'd better to flip it along the 2nd dimension
    matFAOut[marginD[0]:row + marginD[0], col + marginD[1]:matFAOut.shape[1], marginD[2]:leng + marginD[2]] = matFA[:,col - marginD[1]: matFA.shape[1], :]

    # we'd better flip it along the 3rd dimension
    matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matFA[:, :, 0:marginD[2]]
    matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2] + leng:matFAOut.shape[2]] = matFA[:,:, leng - marginD[2]:matFA.shape[2]]


    ## matMR
    # we'd better flip it along the first dimension
    matMROut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMR[0:marginD[0], :, :]
    # we'd better flip it along the 1st dimension
    matMROut[row + marginD[0]:matMROut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMR[row -marginD[0]:matMR.shape[ 0], :,:]

    # we'd better flip it along the 2nd dimension
    matMROut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matMR[:, 0:marginD[1], :]
    # we'd better to flip it along the 2nd dimension
    matMROut[marginD[0]:row + marginD[0], col + marginD[1]:matMROut.shape[1], marginD[2]:leng + marginD[2]] = matMR[:,col-marginD[1]:matMR.shape[1], :]

    # we'd better flip it along the 3rd dimension
    matMROut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matMR[:, :, 0:marginD[2]]
    matMROut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2] + leng:matMROut.shape[2]] = matMR[:,:, leng - marginD[2]:matMR.shape[2]]

    matOut = np.zeros((matSeg.shape[0], matSeg.shape[1], matSeg.shape[2]))
    used = np.zeros((matSeg.shape[0], matSeg.shape[1], matSeg.shape[2])) + eps
    # fid=open('trainxxx_list.txt','a');
    print 'last i ', row - dSeg[0]
    for i in range(0, row - dSeg[0] + 1, step[0]):
        #         print 'i ',i
        for j in range(0, col - dSeg[1] + 1, step[1]):
            for k in range(0, leng - dSeg[2] + 1, step[2]):
                volSeg = matSeg[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                # print 'volSeg shape is ',volSeg.shape
                volFA = matFAOut[i:i + dSeg[0] + 2 * marginD[0], j:j + dSeg[1] + 2 * marginD[1],
                        k:k + dSeg[2] + 2 * marginD[2]]
                volMR = matMROut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                volMat = np.concatenate((volFA, volMR), 0)
                # print 'volFA shape is ',volFA.shape
                # mynet.blobs['dataMR'].data[0,0,...]=volFA
                # mynet.forward()
                # temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                if type=='reg':
                    temppremat = evaluate_reg(volMat, netG, modelPath)
                else:
                    temppremat = evaluate(volMat, netG, modelPath)
                # temppremat = evaluate(volMat, netG, modelPath)
                #                 print 'temppremat shape 1: ',temppremat.shape
                if len(temppremat.shape) == 2:
                    temppremat = np.expand_dims(temppremat, axis=0)
                # print 'patchout shape ',temppremat.shape
                # temppremat=volSeg
                #                 print 'temppremat shape 2: ',temppremat.shape
                matOut[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]] = matOut[i:i + dSeg[0], j:j + dSeg[1],
                                                                      k:k + dSeg[2]] + temppremat;
                used[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]] = used[i:i + dSeg[0], j:j + dSeg[1],
                                                                    k:k + dSeg[2]] + 1;
    matOut = matOut / used
    return matOut


'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use majority voting at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    NumOfClass: number of classes
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def testOneSubject(MR_image, CT_GT, NumOfClass, MR_patch_sz, CT_patch_sz, step, netG, modelPath):
    eps=1e-5
    
    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    [row,col,leng] = matFA.shape
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    
    #print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    #print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

#     matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
#     matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
#     matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    

#     dim1=np.arange(80,192)
#     dim2=np.arange(35,235)
#     x1=80
#     x2=192
#     y1=35
#     y2=235
# #     matFAOutScale = matFAOut[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
# #     matSegScale = matSeg[:,y1:y2,x1:x2]
    matFAOutScale = matFAOut
    matSegScale = matSeg
    matOut = np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass),dtype=np.int32)
    [row,col,leng] = matSegScale.shape
        
    cnt = 0
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOutScale[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                cnt = cnt + 1
                temppremat = evaluate(volFA, netG, modelPath)
#                 volPre = sitk.GetImageFromArray(temppremat)
#                 sitk.WriteImage(volPre,'volPre_{}'.format(cnt)+'.nii.gz')
                
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                for labelInd in range(NumOfClass): #note, start from 0
                    currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                    #scio.savemat('volOut_%d'%cnt+'_label%d.mat'%labelInd,{'currLabelMat%d'%labelInd:currLabelMat})
                    matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd]+currLabelMat;
       
    #scio.savemat('matOut%s.mat'%fileID,{'matOut':matOut})
    matOut = matOut.argmax(axis=3)
    matOut = np.rint(matOut) #this is necessary to convert the data type to be accepted by NIFTI, otherwise will appear strange errors
#     print 'line 378: matOut shape: ',matOut.shape
#     matOut1 = matOut
#     matOut1=np.zeros([matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]])
#     matOut1[:,y1:y2,x1:x2]=matOut
    #matOut1=np.transpose(matOut1,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
    return matOut,matSeg

'''
    used as list in argparse
'''
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
