# medSynthesisV1
This is a copy of package for medical image synthesis work with LRes-ResUnet and GAN (wgan-gp) in pytorch framework, which is an simple extension of our paper <a href='https://ieeexplore.ieee.org/abstract/document/8310638/'>Medical Image Synthesis with Deep Convolutional Adversarial Networks</a>.

# How to run the pytorch code
The main entrance for the code is runCTRecon.py or runCTRecon3d.py

I suppose you have installed:    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pytorch (>=0.3.0)
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;simpleITK 
     <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numpy

Steps to run the code:
1. use extract23DPatch4MultiModalImg.py.py to extract patches (as limited annotated data can be acquired in medical image fields, we usually use patch as the training unit), and save as hdf5 format.
2. choose the generator (1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 4)
3. choose the discriminator if you want to use the GAN-framework (we provide wgan-gp and the basic GAN)
4. choose the loss function (1. LossL1, 2. lossRTL1, 3. MSE (default))
5. set up the hyper-parameters in the runCTRecon.py
6. run the code: python runCTRecon.py

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
