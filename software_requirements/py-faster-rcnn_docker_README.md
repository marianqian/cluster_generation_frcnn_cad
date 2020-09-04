Pull repository to download the image. 

Source: https://github.com/rbgirshick/py-faster-rcnn/issues/509?_pjax=%23js-repo-pjax-container#issuecomment-414398031
Created by @cewee (GitHub): https://gist.github.com/cewee/356b941a4006a502a67f68213f1a76b5 (Dockerfile)

Specifics:
* CUDA 6.0, cuDNN 6, Ubuntu 16.04
* After creating image from the Dockerfile above, this image includes frcnn_cad GitHub repo (https://github.com/riblidezso/frcnn_cad) and py-faster-rcnn GitHub repo (https://github.com/rbgirshick/py-faster-rcnn/). 
* This image also includes the models for py-faster-rcnn downloaded in the /opt/py-faster-rcnn/data folder. Weights can be downloaded `wget https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz` and `tar zxvf faster_rcnn_models.tgz`. If model cannot be downloaded, look through the issues in the py-faster-rcnn GitHub repo. 
* This image also includes the weights downloaded for the frcnn_cad model through `wget http://dkrib.web.elte.hu/cad_faster_rcnn/vgg16_frcnn_cad.caffemodel` in /opt/frcnn_cad folder. 
* Image has Jupyter Notebook installed: `pip install notebook`. 

Instructions of how to create a container can be found here: https://github.com/marianqian/cluster_generation_frcnn_cad. 
