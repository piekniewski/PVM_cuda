### PVM CUDA IMPLEMENTATION

## Introduction
This code is the implementation of Predictive Vision Model first proposed in "Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network" (Piekniewski et al., 2016 (https://arxiv.org/abs/1607.06854)). 
Unlike the original implementation, this version leverages cuda and allows for much faster execution on GPU.

## Working with this code
First you will need to make sure your system is a Linux, has nvidia card and has nvidia-docker installed (or nvidia contrainer runtime). Ubuntu 22.04LTS is recommended as this has been tested on it. Next you will build the container by running a command:
```
bash docker_build.sh
```
This will pull the latest base image and perform a few installs and tweaks. Next depending on what you want to do with this mode, you may need to download original labeled PVM data, a version converted to new format (zip file) can be downloaded from here: http://filip.piekniewski.info/stuff/data/PVM_data.zip

Unzip the data folder in same directory as this code. Next you can run the docker:
```
bash docker_run.sh
```
This command will automatically mount current directory inside the container and install the pvmcuda_pkg python package, and open up a shell session. Now you should be able to see all the contents of the directory containing this repo in /pvm mountpoint. You can now run a test training command, for example:
```
pvm -S model_zoo/small.json -p ./PVM_data/ -d green_ball_training
```
## Debug console
The model once it is executing allows to access some functionality via debug console. You can log into that console on port 9000 but you need to run it from the docker container. Open up another terminal session and check the names of running containers using docker ps command. Next exec into the PVM container:
```
docker exec -it container_name bash
```
Now you are in the same container as the PVM execution. You can type:
```
netcat localhost 9000
```
to access the console. 
