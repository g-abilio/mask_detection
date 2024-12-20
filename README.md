# Real Time Mask Detection

## Introduction 

This repo uses various computer vision technologies, such as deep learning, image acquisition, image processing, feature recognition, and many more, to produce a Real Time Mask Detection application. 

<img src="aux_data/readme_images/with_mask.png" width="400" height="235" alt="Screenshot"> <img src="aux_data/readme_images/no_mask.png" width="400" height="235" alt="Screenshot">

## CNN network structure

The neural net used in this project is based on the CNN developed in the article:
Kim, J.H., Kim, B.G., Roy, P.P. & Jeong, D.M. Efficient facial expression 
recognition algorithm based on hierarchical deep neural network 
structure. IEEE Access 7, 41273–41285 (2019). 

An image, retrieved from the paper, representing this convnet is below:

<img src="aux_data/readme_images/convnet_structure.png" width="600" height="200" alt="Screenshot">

## Instructions for Use

1. Clone the repo: 
```bash
git clone git@github.com:g-abilio/mask_detection.git
```

2. Enter in the folder: 
```bash
cd mask_detection
```

3. Run the following command: 
```bash
chmod +x run.sh
```

4. Run the utility: 
```bash
./run.sh
```
* To quit the real time detection, press the "q" key.

#### Important Observations: 
* The app requires a good ilumination to work correctly.
* Now, the compatibility is only with macOS systems (changes in the future).

### Main technologies used: 
<img src="https://github.com/devicons/devicon/blob/master/icons/pytorch/pytorch-original.svg" title="PyTorch" alt="PyTorch" width="55" height="55"/> <img src="https://github.com/devicons/devicon/blob/master/icons/pandas/pandas-original.svg" title="Pandas" alt="Pandas" width="55" height="55"/> <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original-wordmark.svg" title="Numpy" alt="Numpy" width="55" height="55"/> <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="55" height="55"/>

