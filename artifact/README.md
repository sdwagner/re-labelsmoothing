# How to Run


## Prerequisites

- Nvidia Driver
- Docker
- Nvidia Container Toolkit

## Running the code


Steps to build the containerized enviroment and run the scripts. 

```bash
cd re-labelsmoothing # repository root

# Build the container
docker build -t re_smoothing -f artifact/Dockerfile .

# export the image
docker save re_smoothing > re_smoothing.tar

# Run the container
docker run --gpus all --rm -v ${PWD}:/workspace -it -p 8888:8888 re_smoothing  bash

```

## Inside the Container

```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8888
```
