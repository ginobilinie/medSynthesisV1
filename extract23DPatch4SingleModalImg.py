'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
Created in June, 2016
Author: Dong Nie
'''

import SimpleITK as sitk

from multiprocessing import Pool
import os, argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")

global opt
opt = parser.parse_args()

d1 = 5
d2 = 64
d3 = 64
dFA = [d1, d2, d3]  # size of patches of input data
dSeg = [1, 64, 64]  # size of pathes of label data
step1 = 1
step2 = 16 
step3 = 16
step = [step1, step2, step3]


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix):
                        files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix):
                        files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''


def extractPatch4OneSubject(matFA, matSeg, matMask, fileID, d, step, rate):
    eps = 5e-2
    rate1 = 1.0 / 2
    rate2 = 1.0 / 4
    [row, col, leng] = matFA.shape
    cubicCnt = 0
    estNum = 40000
    trainFA = np.zeros([estNum, 1, dFA[0], dFA[1], dFA[2]], dtype=np.float16)
    trainSeg = np.zeros([estNum, 1, dSeg[0], dSeg[1], dSeg[2]], dtype=np.float16)

    print 'trainFA shape, ', trainFA.shape
    # to padding for input
    margin1 = (dFA[0] - dSeg[0]) / 2
    margin2 = (dFA[1] - dSeg[1]) / 2
    margin3 = (dFA[2] - dSeg[2]) / 2
    cubicCnt = 0
    marginD = [margin1, margin2, margin3]
    print 'matFA shape is ', matFA.shape
    matFAOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]], dtype=np.float16)
    print 'matFAOut shape is ', matFAOut.shape
    matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA

    matSegOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]], dtype=np.float16)
    matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg

    matMaskOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]], dtype=np.float16)
    matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask

    # for mageFA, enlarge it by padding
    if margin1 != 0:
        matFAOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[marginD[0] - 1::-1, :,
                                                                                            :]  # reverse 0:marginD[0]
        matFAOut[row + marginD[0]:matFAOut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[
                                                                                                                  matFA.shape[
                                                                                                                      0] - 1:row -
                                                                                                                             marginD[
                                                                                                                                 0] - 1:-1,
                                                                                                                  :,
                                                                                                                  :]  # we'd better flip it along the 1st dimension
    if margin2 != 0:
        matFAOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matFA[:, marginD[1] - 1::-1,
                                                                                            :]  # we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row + marginD[0], col + marginD[1]:matFAOut.shape[1], marginD[2]:leng + marginD[2]] = matFA[
                                                                                                                  :,
                                                                                                                  matFA.shape[
                                                                                                                      1] - 1:col -
                                                                                                                             marginD[
                                                                                                                                 1] - 1:-1,
                                                                                                                  :]  # we'd flip it along the 2nd dimension
    if margin3 != 0:
        matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matFA[:, :, marginD[
                                                                                                           2] - 1::-1]  # we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2] + leng:matFAOut.shape[2]] = matFA[
                                                                                                                  :, :,
                                                                                                                  matFA.shape[
                                                                                                                      2] - 1:leng -
                                                                                                                             marginD[
                                                                                                                                 2] - 1:-1]
        # for matseg, enlarge it by padding
    if margin1 != 0:
        matSegOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg[marginD[0] - 1::-1,
                                                                                             :,
                                                                                             :]  # reverse 0:marginD[0]
        matSegOut[row + marginD[0]:matSegOut.shape[0], marginD[1]:col + marginD[1],
        marginD[2]:leng + marginD[2]] = matSeg[matSeg.shape[0] - 1:row - marginD[0] - 1:-1, :,
                                        :]  # we'd better flip it along the 1st dimension
    if margin2 != 0:
        matSegOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matSeg[:,
                                                                                             marginD[1] - 1::-1,
                                                                                             :]  # we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row + marginD[0], col + marginD[1]:matSegOut.shape[1],
        marginD[2]:leng + marginD[2]] = matSeg[:, matSeg.shape[1] - 1:col - marginD[1] - 1:-1,
                                        :]  # we'd flip it along the 2nd dimension
    if margin3 != 0:
        matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matSeg[:, :, marginD[
                                                                                                             2] - 1::-1]  # we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1],
        marginD[2] + leng:matSegOut.shape[2]] = matSeg[:, :, matSeg.shape[2] - 1:leng - marginD[2] - 1:-1]

    # for matseg, enlarge it by padding
    if margin1 != 0:
        matMaskOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask[
                                                                                              marginD[0] - 1::-1, :,
                                                                                              :]  # reverse 0:marginD[0]
        matMaskOut[row + marginD[0]:matMaskOut.shape[0], marginD[1]:col + marginD[1],
        marginD[2]:leng + marginD[2]] = matMask[matMask.shape[0] - 1:row - marginD[0] - 1:-1, :,
                                        :]  # we'd better flip it along the 1st dimension
    if margin2 != 0:
        matMaskOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matMask[:,
                                                                                              marginD[1] - 1::-1,
                                                                                              :]  # we'd flip it along the 2nd dimension
        matMaskOut[marginD[0]:row + marginD[0], col + marginD[1]:matMaskOut.shape[1],
        marginD[2]:leng + marginD[2]] = matMask[:, matMask.shape[1] - 1:col - marginD[1] - 1:-1,
                                        :]  # we'd flip it along the 2nd dimension
    if margin3 != 0:
        matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matMask[:, :, marginD[
                                                                                                               2] - 1::-1]  # we'd better flip it along the 3rd dimension
        matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1],
        marginD[2] + leng:matMaskOut.shape[2]] = matMask[:, :, matMask.shape[2] - 1:leng - marginD[2] - 1:-1]

    dsfactor = rate

    for i in range(0, row - dSeg[0], step[0]):
        for j in range(0, col - dSeg[1], step[1]):
            for k in range(0, leng - dSeg[2], step[2]):
                volMask = matMaskOut[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                if np.sum(volMask) < eps:
                    continue
                cubicCnt = cubicCnt + 1
                # index at scale 1
                volSeg = matSeg[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                volFA = matFAOut[i:i + dFA[0], j:j + dFA[1], k:k + dFA[2]]


                trainFA[cubicCnt, 0, :, :, :] = volFA  # 32*32*32

                trainSeg[cubicCnt, 0, :, :, :] = volSeg  # 24*24*24

    trainFA = trainFA[0:cubicCnt, :, :, :, :]
    trainSeg = trainSeg[0:cubicCnt, :, :, :, :]

    with h5py.File('./trainMRCT_snorm_64_%s.h5' % fileID, 'w') as f:
        f['dataMR'] = trainFA
        f['dataCT'] = trainSeg

    with open('./trainMRCT2D_snorm_64_list.txt', 'a') as f:
        f.write('./trainMRCT_snorm_64_%s.h5\n' % fileID)
    return cubicCnt


def main():
    print opt
    path = '/home/niedong/Data4LowDosePET/data_niigz_scale/'
    path = '/shenlab/lab_stor5/dongnie/brain_mr2ct/original_data/'
    scan = ScanFile(path, postfix='_mr.hdr')
    filenames = scan.scan_files()

    maxLPET = 149.366742
    maxPercentLPET = 7.76
    minLPET = 0.00055037
    meanLPET = 0.27593288
    stdLPET = 0.75747500

    # for s-pet
    maxSPET = 156.675962
    maxPercentSPET = 7.79
    minSPET = 0.00055037
    meanSPET = 0.284224789
    stdSPET = 0.7642257

    # for rsCT
    maxCT = 27279
    maxPercentCT = 1320
    minCT = -1023
    meanCT = -601.1929
    stdCT = 475.034

    for filename in filenames:

        print 'low dose filename: ', filename

        lpet_fn = filename
        ct_fn = filename.replace('_mr.hdr', '_ct.hdr')

        imgOrg = sitk.ReadImage(lpet_fn)
        mrnp = sitk.GetArrayFromImage(imgOrg)

        imgOrg1 = sitk.ReadImage(ct_fn)
        ctnp = sitk.GetArrayFromImage(imgOrg1)

        maskimg = mrnp

        mu = np.mean(mrnp)

        if opt.how2normalize == 1:
            maxV, minV = np.percentile(mrnp, [99, 1])
            print 'maxV,', maxV, ' minV, ', minV
            mrnp = (mrnp - mu) / (maxV - minV)
            print 'unique value: ', np.unique(ctnp)

        # for training data in pelvicSeg
        if opt.how2normalize == 2:
            maxV, minV = np.percentile(mrnp, [99, 1])
            print 'maxV,', maxV, ' minV, ', minV
            mrnp = (mrnp - mu) / (maxV - minV)
            print 'unique value: ', np.unique(ctnp)

        # for training data in pelvicSegRegH5
        if opt.how2normalize == 3:
            std = np.std(mrnp)
            mrnp = (mrnp - mu) / std
            print 'maxV,', np.ndarray.max(mrnp), ' minV, ', np.ndarray.min(mrnp)

        if opt.how2normalize == 4:
            maxLPET = 149.366742
            maxPercentLPET = 7.76
            minLPET = 0.00055037
            meanLPET = 0.27593288
            stdLPET = 0.75747500

            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034

            # for s-pet
            maxSPET = 156.675962
            maxPercentSPET = 7.79
            minSPET = 0.00055037
            meanSPET = 0.284224789
            stdSPET = 0.7642257

            # matLPET = (mrnp - meanLPET) / (stdLPET)
            matLPET = (mrnp - minLPET) / (maxPercentLPET - minLPET)
            matCT = (ctnp - meanCT) / stdCT

        if opt.how2normalize == 5:
            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034

            print 'ct, max: ', np.amax(ctnp), ' ct, min: ', np.amin(ctnp)

            # matLPET = (mrnp - meanLPET) / (stdLPET)
            matLPET = mrnp
            matCT = (ctnp - meanCT) / stdCT

        if opt.how2normalize == 6:
            maxPercentPET, minPercentPET = np.percentile(mrnp, [99.5, 0])
            maxPercentCT, minPercentCT = np.percentile(ctnp, [99.5, 0])
            print 'maxPercentMR: ', maxPercentPET, ' minPercentMR: ', minPercentPET, ' maxPercentCT: ', maxPercentCT, 'minPercentCT: ', minPercentCT

            matLPET = (mrnp - minPercentPET) / (maxPercentPET - minPercentPET) #input

            matCT = (ctnp - minPercentPET) / (maxPercentCT - minPercentPET) #output

            print 'maxMR: ', np.amax(matLPET),  ' maxCT: ', np.amax(matCT)
            print 'minLPET: ', np.amin(matLPET),  ' minCT: ', np.amin(matCT)

        # maxV, minV = np.percentile(mrimg, [99.5, 0])
        #         print 'maxV is: ',np.ndarray.max(mrimg)
        #         mrimg[np.where(mrimg>maxV)] = maxV
        #         print 'maxV is: ',np.ndarray.max(mrimg)
        #         mu=np.mean(mrimg) # we should have a fixed std and mean
        #         std = np.std(mrimg)
        #         mrnp = (mrimg - mu)/std
        #         print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)

        # matLPET = (mrimg - meanLPET)/(stdLPET)
        # print 'lpet: maxV,',np.ndarray.max(matLPET),' minV, ',np.ndarray.min(matLPET), ' meanV: ', np.mean(matLPET), ' stdV: ', np.std(matLPET)

        # matLPET = (mrnp - minLPET)/(maxPercentLPET-minLPET)
        # print 'lpet: maxV,',np.ndarray.max(matLPET),' minV, ',np.ndarray.min(matLPET), ' meanV: ', np.mean(matLPET), ' stdV: ', np.std(matLPET)

        #         maxV1, minV1 = np.percentile(mrimg1, [99.5 ,1])
        #         print 'maxV1 is: ',np.ndarray.max(mrimg1)
        #         mrimg1[np.where(mrimg1>maxV1)] = maxV1
        #         print 'maxV1 is: ',np.ndarray.max(mrimg1)
        #         mu1 = np.mean(mrimg1) # we should have a fixed std and mean
        #         std1 = np.std(mrimg1)
        #         mrnp1 = (mrimg1 - mu1)/std1
        #         print 'maxV1,',np.ndarray.max(mrnp1),' minV, ',np.ndarray.min(mrnp1)

        # ctnp[np.where(ctnp>maxPercentCT)] = maxPercentCT
        # matCT = (ctnp - meanCT)/stdCT
        # print 'ct: maxV,',np.ndarray.max(matCT),' minV, ',np.ndarray.min(matCT), 'meanV: ', np.mean(matCT), 'stdV: ', np.std(matCT)

        #         maxVal = np.amax(labelimg)
        #         minVal = np.amin(labelimg)
        #         print 'maxV is: ', maxVal, ' minVal is: ', minVal
        #         mu=np.mean(labelimg) # we should have a fixed std and mean
        #         std = np.std(labelimg)
        #
        #         labelimg = (labelimg - minVal)/(maxVal - minVal)
        #
        #         print 'maxV,',np.ndarray.max(labelimg),' minV, ',np.ndarray.min(labelimg)
        # you can do what you want here for for your label img

        # matSPET = (labelimg - minSPET)/(maxPercentSPET-minSPET)
        # print 'spet: maxV,',np.ndarray.max(matSPET),' minV, ',np.ndarray.min(matSPET), ' meanV: ',np.mean(matSPET), ' stdV: ', np.std(matSPET)

        sdir = filename.split('/')
        print 'sdir is, ', sdir, 'and s6 is, ', sdir[6]
        lpet_fn = sdir[6]
        words = lpet_fn.split('_')
        print 'words are, ', words
        # ind = int(words[0])

        fileID = words[0]
        rate = 1
        cubicCnt = extractPatch4OneSubject(matLPET, matCT, maskimg, fileID, dSeg, step, rate)
        # cubicCnt = extractPatch4OneSubject(mrnp, matCT, hpetnp, maskimg, fileID,dSeg,step,rate)
        print '# of patches is ', cubicCnt

        # reverse along the 1st dimension
        rmrimg = matLPET[matLPET.shape[0] - 1::-1, :, :]
        rmatCT = matCT[matCT.shape[0] - 1::-1, :, :]

        rmaskimg = maskimg[maskimg.shape[0] - 1::-1, :, :]
        fileID = words[0] + 'r'
        cubicCnt = extractPatch4OneSubject(rmrimg, rmatCT, rmaskimg, fileID, dSeg, step, rate)
        print '# of patches is ', cubicCnt


if __name__ == '__main__':
    main()
