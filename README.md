# medSynthesisV1
This is a copy of package for medical image synthesis work with LRes-ResUnet and GAN (wgan-gp) in pytorch framework, which is a simple extension of our paper <a href='https://ieeexplore.ieee.org/abstract/document/8310638/'>Medical Image Synthesis with Deep Convolutional Adversarial Networks</a>. You are also welcome to visit our Tensorflow version through this link:
https://github.com/ginobilinie/medSynthesis

# How to run the pytorch code
The main entrance for the code is runCTRecon.py or runCTRecon3d.py (currently, the 2d/2.5d version is fine to run, and the discriminator for 3d version currently only support BCE loss since I suggest you use W-distance (WGAN-GP) since it is easier to tune the hyper-parameters for this one).

I suppose you have installed:    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pytorch (>=0.3.0)
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;simpleITK 
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numpy

Steps to run the code:
1. use extract23DPatch4MultiModalImg.py (or extract23DPatch4SingleModalImg.py for single input modality) to extract patches for training and validation images (as limited annotated data can be acquired in medical image fields, we usually use patch as the training unit), and save as hdf5 format. Put all these h5 files into two folders (training, validation), and remeber the path to these h5 files
2. choose the generator (1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 4)<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Note: for low-dose xx to standard-dose xx (such as low-dose pet to standard pet, low CT to High CT...) or low resolution xx to high resolution xx(e.g., 3T->7T), we suggest use ResUNet_LRes(4) which contains a long-skip connection. 
 <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If the input modality and the output modality is quite different, we suggest use UNet_LRes(3))
3. choose the discriminator if you want to use the GAN-framework (we provide wgan-gp and the basic GAN)
4. choose the loss function (1. LossL1, 2. lossRTL1, 3. MSE (default))
5. set up the hyper-parameters in the runCTRecon.py (or 3d with runCTRecon3d.py)<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You have to place the paths to the training h5 files (path_patients_h5), the validation h5 files (path_patients_h5_test) and also the path to the testing images (path_test ) in the this python file<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Also, you have to setup all other config choices, such as network choice, disciminator choise, loss functions (including some additional loss, i.e., gradient difference loss), initial learing rate, decrease learning rate during training even with adam optimal solver and so on
6. run the code: python runCTRecon.py (or 3d with runCTRecon3d.py) for training stage
7. run the code: python runTesting_Reconv2.py for testing stage

If it is helpful to your work, please cite the papers:

@inproceedings{nie2017medical,
  title={Medical image synthesis with context-aware generative adversarial networks},
  author={Nie, Dong and Trullo, Roger and Lian, Jun and Petitjean, Caroline and Ruan, Su and Wang, Qian and Shen, Dinggang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={417--425},
  year={2017},
  organization={Springer}
}

@article{nie2018medical,
  title={Medical Image Synthesis with Deep Convolutional Adversarial Networks},
  author={Nie, Dong and Trullo, Roger and Lian, Jun and Wang, Li and Petitjean, Caroline and Ruan, Su and Wang, Qian and Shen, Dinggang},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2018},
  publisher={IEEE}
}

# Dataset
BTW, you can download a real medical image synthesis dataset for reconstructing standard-dose PET from low-dose PET via this link:  https://www.aapm.org/GrandChallenge/LowDoseCT/

Also, there are some MRI synthesis datasets available:
http://brain-development.org/ixi-dataset/

Tumor prediction: 
https://www.med.upenn.edu/sbia/brats2018/data.html

fastMRI:
https://fastmri.med.nyu.edu/

# Upload your brain MRI, Predict corresponding CT

If you're interested in it, you can send me a copy of your data (for example, brain MRI), and I'll inference the CT and send a copy of predicted CT to you. My email is dongnie.at.cs.unc.edu.

# License
medSynthesis is released under the MIT License (refer to the LICENSE file for details).
