# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from Unet3d_pytorch import UNet3D
from nnBuildUnits import CrossEntropy3d, topK_RegLoss, RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
import time
import SimpleITK as sitk

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID", type=int, default=1, help="how to normalize the data")
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=False)
parser.add_argument("--isWDist", action="store_true", help="is adversarial loss with WGAN-GP distance?", default=False)
parser.add_argument("--lambda_AD", default=0.05, type=float, help="weight for AD loss, Default: 0.05")
parser.add_argument("--lambda_D_WGAN_GP", default=10, type=float, help="weight for gradient penalty of WGAN-GP, Default: 10")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--isGDL", action="store_true", help="do we use GDL loss?", default=True)
parser.add_argument("--gdlNorm", default=2, type=int, help="p-norm for the gdl loss, Default: 2")
parser.add_argument("--lambda_gdl", default=0.05, type=float, help="Weight for gdl loss, Default: 0.05")
parser.add_argument("--whichNet", type=int, default=4, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--isMultiSource", action="store_true", help="is multiple modality used?", default=False)
parser.add_argument("--numOfChannel_singleSource", type=int, default=5, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=5, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showValPerformanceEvery", type=int, default=1000, help="number of iterations to show validation performance")
parser.add_argument("--showTestPerformanceEvery", type=int, default=5000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_netD", type=float, default=5e-3, help="Learning Rate for discriminator. Default=5e-3")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="prob to drop neurons to zero: 0.2")
parser.add_argument("--decLREvery", type=int, default=10000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--RT_th", default=0.005, type=float, help="Relative thresholding: 0.005")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="/home/niedong/Data4LowDosePET/pytorch_UNet/resunet2d_dp_pet_BatchAug_sNorm_lres_bn_lr5e3_lrdec_base1_lossL1_lossGDL0p05_0705_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub1_pet_BatchAug_sNorm_resunet_dp_lres_bn_lr5e3_lrdec_base1_lossL1_lossGDL0p05_0705_", type=str, help="prefix of the to-be-saved predicted filename")

global opt, model 
opt = parser.parse_args()

def main():    
    print opt    
        
#     prefixModelName = 'Regressor_1112_'
#     prefixPredictedFN = 'preSub1_1112_'
#     showTrainLossEvery = 100
#     lr = 1e-4
#     showTestPerformanceEvery = 2000
#     saveModelEvery = 2000
#     decLREvery = 40000
#     numofIters = 200000
#     how2normalize = 0


    netD = Discriminator()
    netD.apply(weights_init)
    netD.cuda()
    
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lr_netD)
    criterion_bce=nn.BCELoss()
    criterion_bce.cuda()
    
    #net=UNet()
    if opt.whichNet==1:
        net = UNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==2:
        net = ResUNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==3:
        net = UNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==4:
        net = ResUNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1, dp_prob = opt.dropout_rate)
    #net.apply(weights_init)
    net.cuda()
    params = list(net.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
 
    
    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_RTL1 = RelativeThreshold_RegLoss(opt.RT_th)
    criterion_gdl = gdl_loss(opt.gdlNorm)
    #criterion = nn.CrossEntropyLoss()
#     criterion = nn.NLLLoss2d()
    
    given_weight = torch.cuda.FloatTensor([1,4,4,2])
    
    criterion_3d = CrossEntropy3d(weight=given_weight)
    
    criterion_3d = criterion_3d.cuda()
    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    
    #inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
    #targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    
    path_test ='/home/niedong/DataCT/data_niigz/'
    path_patients_h5 = '/home/niedong/DataCT/h5Data_snorm/trainBatch2D_H5'
    path_patients_h5_val ='/home/niedong/DataCT/h5Data_snorm/valBatch2D_H5'
#     batch_size=10
    #data_generator = Generator_2D_slices(path_patients_h5,opt.batchSize,inputKey='data3T',outputKey='data7T')
    #data_generator_test = Generator_2D_slices(path_patients_h5_test,opt.batchSize,inputKey='data3T',outputKey='data7T')

    data_generator = Generator_2D_slicesV1(path_patients_h5,opt.batchSize, inputKey='dataLPET', segKey='dataCT', contourKey='dataHPET')
    data_generator_test = Generator_2D_slicesV1(path_patients_h5_val, opt.batchSize, inputKey='dataLPET', segKey='dataCT', contourKey='dataHPET')
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['model'])
            opt.start_epoch = 100000
            opt.start_epoch = checkpoint["epoch"] + 1
            # net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        #print('iter %d'%iter)
        
        inputs, exinputs, labels = data_generator.next()
#         xx = np.transpose(inputs,(5,64,64))
#         inputs = np.transpose(inputs,(0,3,1,2))
        inputs = np.squeeze(inputs) #5x64x64
        # exinputs = np.transpose(exinputs,(0,3,1,2))
        exinputs = np.squeeze(exinputs) #5x64x64
#         print 'shape is ....',inputs.shape
        labels = np.squeeze(labels) #64x64
#         labels = labels.astype(int)

        inputs = inputs.astype(float)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        exinputs = exinputs.astype(float)
        exinputs = torch.from_numpy(exinputs)
        exinputs = exinputs.float()
        labels = labels.astype(float)
        labels = torch.from_numpy(labels)
        labels = labels.float()
        #print type(inputs), type(exinputs)
        if opt.isMultiSource:
            source = torch.cat((inputs, exinputs),dim=1)
        else:
            source = inputs
        #source = inputs
        mid_slice = opt.numOfChannel_singleSource//2
        residual_source = inputs[:, mid_slice, ...]
        #inputs = inputs.cuda()
        #exinputs = exinputs.cuda()
        source = source.cuda()
        residual_source = residual_source.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        source, residual_source, labels = Variable(source),Variable(residual_source), Variable(labels)
        #inputs, exinputs, labels = Variable(inputs),Variable(exinputs), Variable(labels)
        
        ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
        if opt.isAdLoss:
            outputG = net(source,residual_source) #5x64x64->1*64x64
            
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)
            netD.zero_grad()
            batch_size = inputs.size(0)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()
            #print(real_label.size())
            real_label = Variable(real_label)
            #print(outputD_real.size())
            loss_real = criterion_bce(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label = torch.zeros(batch_size,1)
    #         fake_label = torch.FloatTensor(batch_size)
    #         fake_label.data.resize_(batch_size).fill_(0)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake,fake_label)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake
#             print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',outputD_real.data[0]
#             print('loss for discriminator is %f'%lossD.data[0])
            #update network parameters
            optimizerD.step()
            
        if opt.isWDist:
            one = torch.FloatTensor([1])
            mone = one * -1
            one = one.cuda()
            mone = mone.cuda()
            
            netD.zero_grad()
            
            outputG = net(source,residual_source) #5x64x64->1*64x64
            
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)

            
            batch_size = inputs.size(0)
            
            D_real = outputD_real.mean()
            # print D_real
            D_real.backward(mone)
        
        
            D_fake = outputD_fake.mean()
            D_fake.backward(one)
        
            gradient_penalty = opt.lambda_D_WGAN_GP*calc_gradient_penalty(netD, labels.data, outputG.data)
            gradient_penalty.backward()
            
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            
            optimizerD.step()
        
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
#         print inputs.data.shape
        #outputG = net(source) #here I am not sure whether we should use twice or not
        outputG = net(source,residual_source) #5x64x64->1*64x64
        net.zero_grad()
        if opt.whichLoss==1:
            lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
        elif opt.whichLoss==2:
            lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
        else:
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
        lossG_G = opt.lossBase * lossG_G
        lossG_G.backward() #compute gradients

        if opt.isGDL:
            lossG_gdl = opt.lambda_gdl * criterion_gdl(outputG,torch.unsqueeze(labels,1))
            lossG_gdl.backward() #compute gradients

        if opt.isAdLoss:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            #outputG = net(inputs)
            outputG = net(source,residual_source) #5x64x64->1*64x64
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD = netD(outputG)
            lossG_D = opt.lambda_AD*criterion_bce(outputD,real_label) #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward()
            
        if opt.isWDist:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            #outputG = net(inputs)
            outputG = net(source,residual_source) #5x64x64->1*64x64
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD_fake = netD(outputG)

            outputD_fake = outputD_fake.mean()
            
            lossG_D = opt.lambda_AD*outputD_fake.mean() #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward(mone)
        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizer.step() #update network parameters

        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
        running_loss = running_loss + lossG_G.data[0]

        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print '************************************************'
            print 'time now is: ' + time.asctime(time.localtime(time.time()))
#             print 'running loss is ',running_loss
            print 'average running loss for generator between iter [%d, %d] is: %.5f'%(iter - 100 + 1,iter,running_loss/100)
            
            print 'lossG_G is %.5f respectively.'%(lossG_G.data[0])

            if opt.isGDL:
                print('loss for GDL loss is %f'%lossG_gdl.data[0])

            if opt.isAdLoss:
                print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',outputD_real.data[0]
                print('loss for discriminator is %f'%lossD.data[0])  
                
            if opt.isWDist:
                print 'loss_real is ',D_real.data[0],'loss_fake is ',D_fake.data[0]
                print('loss for discriminator is %f'%Wasserstein_D.data[0], ' D cost is %f'%D_cost)                
            
            print 'cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start)
            print '************************************************'
            running_loss = 0.0
            start = time.time()
        if iter%opt.saveModelEvery==0: #save the model
            state = {
                'epoch': iter+1,
                'model': net.state_dict()
            }
            torch.save(state, opt.prefixModelName+'%d.pt'%iter)
            print 'save model: '+opt.prefixModelName+'%d.pt'%iter

            if opt.isAdLoss:
                torch.save(netD.state_dict(), opt.prefixModelName+'_net_D%d.pt'%iter)
        if iter%opt.decLREvery==0:
            opt.lr = opt.lr*0.5
            adjust_learning_rate(optimizer, opt.lr)
                
        if iter%opt.showValPerformanceEvery==0: #test one subject
            # to test on the validation dataset in the format of h5 
            inputs,exinputs,labels = data_generator_test.next()

            # inputs = np.transpose(inputs,(0,3,1,2))
            inputs = np.squeeze(inputs)

            # exinputs = np.transpose(exinputs, (0, 3, 1, 2))
            exinputs = np.squeeze(exinputs)  # 5x64x64

            labels = np.squeeze(labels)

            inputs = torch.from_numpy(inputs)
            inputs = inputs.float()
            exinputs = torch.from_numpy(exinputs)
            exinputs = exinputs.float()
            labels = torch.from_numpy(labels)
            labels = labels.float()
            mid_slice = opt.numOfChannel_singleSource // 2
            residual_source = inputs[:, mid_slice, ...]
            if opt.isMultiSource:
                source = torch.cat((inputs, exinputs), dim=1)
            else:
                source = inputs
            source = source.cuda()
            residual_source = residual_source.cuda()
            labels = labels.cuda()
            source,residual_source,labels = Variable(source),Variable(residual_source), Variable(labels)

            # source = inputs
            #outputG = net(inputs)
            outputG = net(source,residual_source) #5x64x64->1*64x64

            if opt.whichLoss == 1:
                lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
            elif opt.whichLoss == 2:
                lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
            else:
                lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
            lossG_G = opt.lossBase * lossG_G
            print '.......come to validation stage: iter {}'.format(iter),'........'
            print 'lossG_G is %.5f.'%(lossG_G.data[0])

            if opt.isGDL:
                lossG_gdl = criterion_gdl(outputG, labels)
                print('loss for GDL loss is %f'%lossG_gdl.data[0])

        if iter % opt.showTestPerformanceEvery == 0:  # test one subject
            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'sub1_sourceCT.nii.gz'))
            ct_test_itk=sitk.ReadImage(os.path.join(path_test,'sub1_extraCT.nii.gz'))
            hpet_test_itk = sitk.ReadImage(os.path.join(path_test, 'sub1_targetCT.nii.gz'))

            spacing = hpet_test_itk.GetSpacing()
            origin = hpet_test_itk.GetOrigin()
            direction = hpet_test_itk.GetDirection()

            mrnp=sitk.GetArrayFromImage(mr_test_itk)
            ctnp=sitk.GetArrayFromImage(ct_test_itk)
            hpetnp=sitk.GetArrayFromImage(hpet_test_itk)

            ##### specific normalization #####
            # mu = np.mean(mrnp)
            # maxV, minV = np.percentile(mrnp, [99 ,25])
            # #mrimg=mrimg
            # mrnp = (mrnp-minV)/(maxV-minV)



            #for training data in pelvicSeg
            if opt.how2normalize == 1:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp = (mrnp-mu)/(maxV-minV)
                print 'unique value: ',np.unique(ctnp)

            #for training data in pelvicSeg
            if opt.how2normalize == 2:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp = (mrnp-mu)/(maxV-minV)
                print 'unique value: ',np.unique(ctnp)

            #for training data in pelvicSegRegH5
            if opt.how2normalize== 3:
                std = np.std(mrnp)
                mrnp = (mrnp - mu)/std
                print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)

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

                #matLPET = (mrnp - meanLPET) / (stdLPET)
                matLPET = (mrnp - minLPET) / (maxPercentLPET - minLPET)
                matCT = (ctnp - meanCT) / stdCT
                matSPET = (hpetnp - minSPET) / (maxPercentSPET - minSPET)

            if opt.how2normalize == 5:
                # for rsCT
                maxCT = 27279
                maxPercentCT = 1320
                minCT = -1023
                meanCT = -601.1929
                stdCT = 475.034

                print
                'ct, max: ', np.amax(ctnp), ' ct, min: ', np.amin(ctnp)

                # matLPET = (mrnp - meanLPET) / (stdLPET)
                matLPET = mrnp
                matCT = (ctnp - meanCT) / stdCT
                matSPET = hpetnp

            if opt.how2normalize == 6:
                maxPercentPET, minPercentPET = np.percentile(mrnp, [99.5, 0])
                maxPercentCT, minPercentCT = np.percentile(ctnp, [99.5, 0])
                print 'maxPercentPET: ', maxPercentPET, ' minPercentPET: ', minPercentPET, ' maxPercentCT: ', maxPercentCT, 'minPercentCT: ', minPercentCT

                matLPET = (mrnp - minPercentPET) / (maxPercentPET - minPercentPET)
                matSPET = (hpetnp - minPercentPET) / (maxPercentPET - minPercentPET)

                matCT = (ctnp - minPercentCT) / (maxPercentCT - minPercentCT)


            if not opt.isMultiSource:
                matFA = matLPET
                matGT = hpetnp

                print 'matFA shape: ',matFA.shape, ' matGT shape: ', matGT.shape
                matOut = testOneSubject_aver_res(matFA,matGT,[5,64,64],[1,64,64],[1,32,32],net,opt.prefixModelName+'%d.pt'%iter)
                print 'matOut shape: ',matOut.shape
                if opt.how2normalize==6:
                    ct_estimated = matOut * (maxPercentPET - minPercentPET) + minPercentPET
                else:
                    ct_estimated = matOut


                itspsnr = psnr(ct_estimated, matGT)

                print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
                print 'gt: ',ctnp.dtype,' shape: ',ct_estimated.shape
                print 'psnr = ',itspsnr
                volout = sitk.GetImageFromArray(ct_estimated)
                volout.SetSpacing(spacing)
                volout.SetOrigin(origin)
                volout.SetDirection(direction)
                sitk.WriteImage(volout,opt.prefixPredictedFN+'{}'.format(iter)+'.nii.gz')
            else:
                matFA = matLPET
                matGT = hpetnp
                print 'matFA shape: ', matFA.shape, ' matGT shape: ', matGT.shape
                matOut = testOneSubject_aver_res_multiModal(matFA, matCT, matGT, [5, 64, 64], [1, 64, 64], [1, 32, 32], net,
                                                            opt.prefixModelName + '%d.pt' % iter)
                print 'matOut shape: ', matOut.shape
                if opt.how2normalize==6:
                    ct_estimated = matOut * (maxPercentPET - minPercentPET) + minPercentPET
                else:
                    ct_estimated = matOut

                itspsnr = psnr(ct_estimated, matGT)

                print 'pred: ', ct_estimated.dtype, ' shape: ', ct_estimated.shape
                print 'gt: ', ctnp.dtype, ' shape: ', ct_estimated.shape
                print 'psnr = ', itspsnr
                volout = sitk.GetImageFromArray(ct_estimated)
                volout.SetSpacing(spacing)
                volout.SetOrigin(origin)
                volout.SetDirection(direction)
                sitk.WriteImage(volout, opt.prefixPredictedFN + '{}'.format(iter) + '.nii.gz')
        
    print('Finished Training')
    
if __name__ == '__main__':   
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)  
    main()
    
