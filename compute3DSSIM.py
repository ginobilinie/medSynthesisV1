'''
Target: Compute structure similarity (SSIM) between two 3D volumes 
Created on Jan, 22th 2018
Author: Dong Nie

reference from: http://simpleitk-prototype.readthedocs.io/en/latest/user_guide/plot_image.html
'''

import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np
import scipy.io as scio
from morpologicalTransformation import denoiseImg_closing, denoiseImg_isolation
from skimage import measure


path = '/shenlab/lab_stor5/dongnie/3T7T/results/'


def main():
    ids = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    ids = range(0, 30)
    ids = [1]
    for id in ids:
        #         datafn = os.path.join(path,'Case%02d.mhd'%id)
        #         outdatafn = os.path.join(path,'Case%02d.nii.gz'%id)
        #
        #         dataOrg = sitk.ReadImage(datafn)
        #         dataMat = sitk.GetArrayFromImage(dataOrg)
        #         #gtMat=np.transpose(gtMat,(2,1,0))
        #         dataVol = sitk.GetImageFromArray(dataMat)
        #         sitk.WriteImage(dataVol,outdatafn)

        # datafn = os.path.join(path, 'img1.mhd')
        # dataOrg = sitk.ReadImage(datafn)
        # spacing = dataOrg.GetSpacing()
        # origin = dataOrg.GetOrigin()
        # direction = dataOrg.GetDirection()
        # dataMat = sitk.GetArrayFromImage(dataOrg)

        gtfn = os.path.join(path, 'S1to1_7t.nii.gz')
        gtOrg = sitk.ReadImage(gtfn)
        gtMat = sitk.GetArrayFromImage(gtOrg)
        # gtMat=np.transpose(gtMat,(2,1,0))
        gtMat = gtMat.astype(np.float32)
        gtMat = (gtMat - np.amin(gtMat))/(np.amax(gtMat)-np.amin(gtMat))
        print np.amax(gtMat),',', np.amin(gtMat), 'dtype: ',gtMat.dtype, ',',gtMat.shape

        prefn = os.path.join(path,'preSub1_1112_195000.nii.gz')
        preOrg = sitk.ReadImage(prefn)
        preMat = sitk.GetArrayFromImage(preOrg)
        preMat = preMat.astype(np.float32)
        preMat = (preMat - np.amin(preMat))/(np.amax(preMat)-np.amin(preMat))
        # preMat = np.transpose(preMat,(2,0,1))
        print np.amax(preMat),',', np.amin(preMat), 'dtype:',preMat.dtype, ',',preMat.shape


        ssim_3d_sk = measure.compare_ssim(gtMat, preMat, multichannel=True, gaussian_weights=True, data_range=1.0,
                                          use_sample_covariance=False)
        print ssim_3d_sk

        # ssim_3d_sk = measure.structural_similarity(gtMat, preMat, multichannel=True, gaussian_weights=True, data_range=1.0,
        #                                   use_sample_covariance=False)
        # gtMat1 = denoiseImg_closing(gtMat, kernel=np.ones((20, 20, 20)))
        # gtMat2 = gtMat + gtMat1
        # gtMat2[np.where(gtMat2 > 1)] = 1
        # gtMat = gtMat2
        # gtMat = denoiseImg_isolation(gtMat, struct=np.ones((3, 3, 3)))
        #
        # gtMat = gtMat.astype(np.uint8)

        # ind1 = np.where((gtMat==1)&(preMat==1))
        # preMat[ind1] = 0
        # ind2 = np.where((gtMat==2)&(preMat==2))
        # preMat[ind2] = 0
        # ind3 = np.where((gtMat==3)&(preMat==3))
        # preMat[ind3] = 0
        # errorMat = preMat
        #
        # outgtfn = os.path.join(path, 'sgm_errormap_sub1.nii.gz')
        # errorVol = sitk.GetImageFromArray(errorMat)
        # errorVol.SetSpacing(spacing)
        # errorVol.SetOrigin(origin)
        # errorVol.SetDirection(direction)
        # sitk.WriteImage(errorVol, outgtfn)


#
#         prefn='preSub%d_as32_v12.nii'%id
#         preOrg=sitk.ReadImage(prefn)
#         preMat=sitk.GetArrayFromImage(preOrg)
#         preMat=np.transpose(preMat,(2,1,0))
#         preVol=sitk.GetImageFromArra(preMat)
#         sitk.WriteImage(preVol,prefn)


if __name__ == '__main__':
    main()
