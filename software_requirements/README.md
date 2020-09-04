README for using how to use CAD software with MC-GPU mammogram data

I used Docker two different docker containers to run MC-GPU and the CAD software since they had different dependencies with Python (the model for the CAD software uses Python 2). 
Relevant GitHub/Docker Resources: 
-	GiHub repo from the Ribli et. al paper: https://github.com/riblidezso/frcnn_cad
-	GitHub repo for Faster-RCNN network: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/test.py
-	Dockerfile setting up Faster-RCNN software: https://gist.github.com/cewee/356b941a4006a502a67f68213f1a76b5 
-	Quickstart to Docker: https://docs.docker.com/get-started/

Glossary:
-	Docker image: provides a file system and includes everything to run an application (with code, binaries, dependencies etc.) 
-	Docker container: a running process of a Docker image (we can create different version of Docker containers from one Docker image). 

Source for py-faster-rcnn Docker image: https://github.com/rbgirshick/py-faster-rcnn/issues/509?_pjax=%23js-repo-pjax-container

1.	Install Docker on your computer. 

2.	Make sure your computer has the right dependencies for running NVIDIA GPU with a driver since the CAD algorithm requires a GPU (could possibly be used with only a CPU but I haven’t tried that out.)  

3.	Run nvidia-smi to test whether GPU is working correctly. 

I pushed Docker images to my public repo on Docker hub which includes the dependencies with the CAD algorithm and MC-GPU, but below also lists the steps of how I created the containers. 

4.	Download the CAD algorithm image. (Link to repo: https://hub.docker.com/r/mqian36/py-faster-rcnn)
  
    a.	docker pull mqian36/py-faster-rcnn

5.	Run the Docker image you just created as a Docker container. 

    a.	docker run -it --gpus all -v /home:/home -p 8008:8008 --name <INSERT NAME OF CONTAINER> <NAME OF DOCKER IMAGE>:latest /bin/bash
  
    b.	- - gpus is where we can specify the number of GPUS, and allow us to use GPUs inside the Docker container. 

    c.	-v (volumes) allow us to access the data/images we want to pass through the CAD software. Now, you will be able to access the /home folder and all of the files inside while you are also inside the Docker container.

    d.	The port 8008 is what we will use to communicate the docker container and our host computer. Any port that’s not being used would work. From https://docs.docker.com/get-started/part2/: --publish (or - -p) asks Docker to forward traffic incoming on the host’s port 8000 to the container’s port 8080. Containers have their own private set of ports, so if you want to reach one from the network, you have to forward traffic to it in this way. Otherwise, firewall rules will prevent all network traffic from reaching your container, as a default security posture.

6.	Once the Docker container is created, enter the container by running the command below.

    a.	docker exec -it <CONTAINER ID> /bin/bash

    b.	The CONTAINER ID can be found by running docker ps and finding the CONTAINER ID from the image with the Docker image name you choose. 

7.	Once inside the container, the frcnn_cad and py-faster-rcnn GitHub repos will be located in the /opt folder. 

8.	 Run Jupyter notebook and access the demo notebook in your localhost computer’s browser by running the command below. For the host computer, make sure to use the port that was indicated when creating the container.

    a.	jupyter notebook --ip 0.0.0.0 --port 8008 --no-browser --allow-root 

9.	The output after running the command should have links that allow you to access the Jupyter notebook file system on your computer’s browser. 
  
    a.	Another way: On your local host computer, go the url localhost:8008 (or whichever port used when creating the container).  You should be prompted to enter a password or token ID, and that can be found in the output in the terminal after running the command in #12. 

10.	Now you can access the demo notebook which contains the CAD software. Run the cells in order, and the last cell ‘Load and analyze image’ allows you to insert your own image to detect malignant masses or microcalcification clusters by changing the file path to the image in the method cv2.imread(<INSERT PATH OF IMAGE>). Usually, the file path to the image is through the shared folder indicated when the Docker container was first created. 

Creating the MCGPU docker container is through the same way. (Link to repo: https://hub.docker.com/r/mqian36/mcgpu). You would run docker pull mqian36/mcgpu instead to download the image. 

11.	How I created the CAD software container: 

    a.  Create a file named ‘Dockerfile’ with no file extension with the information in https://gist.github.com/cewee/356b941a4006a502a67f68213f1a76b5 inside. The file should be created in its own folder. The Dockerfile creates the Docker image that we will use to run our CAD software. Edit the first line of the Dockerfile so that it works with the systems of your GPU and NVIDIA driver. (I’m not sure if changing these from the original Dockerfile will mess up its dependencies with the actual CAD software).
    
    b.	Create the Docker image by running the command below. I named the Docker image py-faster-rcnn.

        i.	docker build - -tag <NAME OF DOCKER IMAGE> .

    c.	Once inside the container, you should be the root user and inside the folder /opt/py-faster-rcnn. You need to download the models by downloading the Faster-RCNN models.

        i.	cd data

        ii.	wget https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz

        iii.	tar zxvf faster_rcnn_models.tgz
      
    d.	Navigate to the /opt folder and download the GitHub repo with the demo Jupyter notebook. 

        i.	git clone https://github.com/riblidezso/frcnn_cad.git
      
    e.	Download the weights for the model into the /opt/frcnn_cad folder.
  
       i.	Weights can be downloaded through this link: 

    f.	Install Jupyter notebook.
  
        i.	pip install notebook

    g.	After launching Jupyter notebook, change the first line in demo.ipynb in the frcnn_cad repo to /opt/py-faster-rcnn/tools.
