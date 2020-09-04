Pull repository to download the image. 

Docker image with dependencies to run Monte-Carlo GPU simulation software: MC-GPU_v1.5b: VICTRE pivotal study simulations (https://github.com/DIDSR/VICTRE_MCGPU). Image does not have the MC-GPU software files.

Specifics:
* CUDA 10.2, cuDNN 7, Ubuntu 18.04
* Installed OpenMPI library. 
```
apt-get install openmpi
apt-get install openmpi-dev
apt-get install openmpi-bin
apt-get install libopenmpi-dev
```

Instructions of how to create a container can be found here: https://github.com/marianqian/cluster_generation_frcnn_cad. 
