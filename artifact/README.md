# How to Run

Steps to build the containerized enviroment and run the scripts. 

```bash
cd re-labelsmoothing # repository root

docker build -t re_smoothing -f artifact/dockerfile .

docker run --gpus all --rm -v ${PWD}:/workspace -it -p 8888:8888 re_smoothing  bash

```


## Inside the Container


```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8888


```
