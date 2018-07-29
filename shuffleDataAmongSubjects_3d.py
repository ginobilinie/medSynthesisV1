import numpy as np
import h5py
import os

'''
    Shuffle data (patches) among the subjects
input:
    save_dir: the h5 files you save
    num: the num of components in the h5 file
output:
    save as the same name h5 files
'''

# dFA = [3,168,112] # size of patches of input data
# dSeg = [1,168,112] # size of pathes of label data

dFA = [16, 64, 64]  # size of patches of input data
dSeg = [16, 64, 64]  # size of pathes of label data


def shuffleDataAmongSubjects(save_dir, savepath):
    #     allfilenames = os.listdir(save_dir)
    #     allfilenames = filter(lambda x: '.h5' in x and 'train' in x, allfilenames)

    nn = 10000
    dataLPET = np.zeros([nn,1, 16, 64, 64], dtype=np.float16)
    dataCT = np.zeros([nn,1, 16, 64, 64], dtype=np.float16)
    dataHPET = np.zeros([nn, 1, 16, 64, 64], dtype=np.float16)
    #

    allfilenames = os.listdir(save_dir)
    #     print allfilenames
    allfilenames = filter(lambda x: '.h5' in x and 'train' in x, allfilenames)
    #     print allfilenames
    cnt = 0
    numInOneSub = 5
    batchID = 0
    startInd = 0

    savefilename = 'train5x64x64_'
    for i_file, filename in enumerate(allfilenames):

        with h5py.File(os.path.join(save_dir, filename), 'r+') as h5f:
            print '*******path is ', os.path.join(save_dir, filename)
            dLPET = h5f['dataLPET'][:]
            dCT = h5f['dataCT'][:]
            dHPET = h5f['dataHPET'][:]

            unitNum = dLPET.shape[0]
            print 'unitNum: ', unitNum, 'dLPET shape: ', dLPET.shape

            dataLPET[startInd: (startInd + unitNum), ...] = dLPET
            dataCT[startInd: startInd + unitNum, ...] = dCT
            dataHPET[startInd: startInd + unitNum, ...] = dHPET

            startInd = startInd + unitNum

            cnt = cnt + 1

            if cnt == numInOneSub:
                batchID = batchID + 1
                dataLPET = dataLPET[0:startInd, ...]
                dataCT = dataCT[0:startInd, ...]
                dataHPET = dataHPET[0:startInd, ...]

                with h5py.File(os.path.join(savepath, savefilename + '{}.h5'.format(batchID)), 'w') as hf:
                    hf.create_dataset('dataLPET', data=dataLPET)
                    hf.create_dataset('dataCT', data=dataCT)
                    hf.create_dataset('dataHPET', data=dataHPET)

                ############ initialization ###############
                cnt = 0
                startInd = 0
                print
                'nn:', nn
                dataLPET = np.zeros([nn, 1, 16, 64, 64], dtype=np.float16)
                dataCT = np.zeros([nn, 1, 16, 64, 64], dtype=np.float16)
                dataHPET = np.zeros([nn, 16, 1, 64, 64], dtype=np.float16)

    #         mean_train, std_train = 0., 0
    batchID = batchID + 1
    if startInd != 0:
        dataLPET = dataLPET[0:startInd, ...]
        dataCT = dataCT[0:startInd, ...]
        dataHPET = dataHPET[0:startInd, ...]

        with h5py.File(os.path.join(savepath, savefilename + '{}.h5'.format(batchID)), 'w') as hf:
            hf.create_dataset('dataLPET', data=dataLPET)
            hf.create_dataset('dataCT', data=dataCT)
            hf.create_dataset('dataHPET', data=dataHPET)

    return


def main():
    path = '/home/niedong/Data4LowDosePET/h5Data3D_noNorm/train3D_H5/'
    savepath = '/home/niedong/Data4LowDosePET/h5DataAug_noNorm/trainBatch3D_H5/'
    basePath = '/home/niedong/Data4LowDosePET/h5DataAug_noNorm/'
    # path = basePath + 'train2D_H5/'
    # savepath = basePath + 'trainBatch2D_H5/'
    shuffleDataAmongSubjects(path, savepath)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
