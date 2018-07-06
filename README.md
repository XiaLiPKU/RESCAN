# RESCAN: Recurrent Squeeze-and-Excitation Context Aggregation Net
Xia Li, Jianlong Wu, Zhouchen Lin, Hong Liu, Hongbin Zha
Peking University, Shenzhen
Peking University, Beijing

Rain streaks can severely degrade the visibility, which causes many current computer vision algorithms fail to work. So it is necessary to remove the rain from images. We propose a novel deep network architecture based on deep convolutional and recurrent neural networks for single image deraining. As contextual information is very important for rain removal, we first adopt the dilated convolutional neural network to acquire large receptive field. To better fit the rain removal task, we also modify the network. In heavy rain, rain streaks have various directions and shapes, which can be regarded as the accumulation of multiple rain streak layers. We assign different alpha-values to various rain streak layers according to the intensity and transparency by incorporating the squeeze-and-excitation block. Since rain streak layers overlap with each other, it is not easy to remove the rain in one stage. So we further decompose the rain removal into multiple stages. Recurrent neural network is incorporated to preserve the useful information in previous stages and benefit the rain removal in later stages. We conduct extensive experiments on both synthetic and real-world datasets. Our proposed method outperforms the state-of-the-art approaches under all evaluation metrics.

## Prerequisite
- Python3.6 # wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh | bash
- Pytorch # pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
- Opencv # conda install -c conda-forge opencv
- tensorboard-pytorch # pip install tensorboardX

## Project Structure
- config: contains all codes
    - cal_ssim.py
    - clean.sh
    - dataset.py
    - main.py
    - model.py
    - settings.py
    - show.py
    - tensorboard.sh
- explore.sh
- logdir: holds patches generated in training process1
- models: holds checkpoints
- showdir: holds images predicted by the model

## Best Practices
Hold every experiment in an independent folder, and assign a long name to it.
We recommand list the important parameters in the folder name, for example: RESCAN.ConvRNN.Full.d_7.c_24(d: depth, c: channel).

## Default Dataset settings
Rain100H: http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
Rain800: https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s

We concat the two images(B and O) together as default inputs. If you want to change this setting, just modify config/dataset.py.
Moreover, there should be three folders 'train', 'val', 'test' in the dataset folder.
After download the datasets, don't forget to transform the format!

## Train, Test and Show
    python main.py -a train
    python main.py -a test
    python show.py

## Scripts
- explore.sh: Show the predicted images in browser
- config/tensorboard.sh: Open the tensorboard server
- config/clean.sh: Clear all the training records in the folder
