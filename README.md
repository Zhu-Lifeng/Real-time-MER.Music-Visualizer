<h1 align='center'>Real-time MER Music Visualizer Framework</h1>

# Architecture 
This framework invokes Cloud Storage and FireStore for back stage database construction and user account management. The framework itself is developed based on Flask.
<p align='center'>
  <img width='700' src="https://github.com/Zhu-Lifeng/Real-time-MER.Music-Visualizer/blob/main/readme/Cloud Architecture.png" alt="Architecture">
</p>

# Project file instruction
The MER model folder contains the file about the Music Emotion Recognizing model this project uses. It needs to be noticed that we use DEAM dataset as the orignal data source so it is required to be stored in the environment for the model related files to be used.

This project is designed to be deploy on GCP, Server-based and Server-less deployment strategy are both available by using the Docker to build a Container to carry the service. Google Cloud Storage and FireStore are employeed so that the code about these two function requires to be self-modified.

# How to deploy

## 1. For sever-based deployment
### 1) have a GCP Compute Engine instance.
### 2) login the instance with the SSH.
### 3) finish the initial setting and installation.
```sh
sudo apt update
sudo apt install python3-pip
sudo apt install git
sudo apt install docker.io
```
### 4) create a virtual environment and login it
```sh
sudo apt install python3.11-venv
python3 -m venv myenv
source myenv/bin/activate
```
### 5) download the application files from git and enter its folder<br>

```sh
git clone https://github.com/Zhu-Lifeng/Real-time-MER.Music-Visualizer.git
cd Real-time-MER.Music-Visualizer
```
### 6) create the container and deploy it
```sh
sudo docker build -t myapp .
sudo docker run -p 8080:8080 myapp
```
