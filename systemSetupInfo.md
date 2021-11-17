# Current Setup Information:

## this setup uses Conda, Ubuntu 20.04 and an RTX3080

## NVIDIA-SMI 460.39       Driver Version: 460.39       CUDA Version: 11.2
### note nvcc displays nothing installed because the apt version of cuda is still stuck at 10
### these are the package versions that have been installed in a special conda environment
#### pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#### cant remember if i pip installed big-sleep or not try without first