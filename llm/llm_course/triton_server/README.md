
```bash
# 准备
mkdir ft_workspace
cd ft_workspace
git clone https://github.com/Rayrtfr/fastertransformer_backend.git
git clone https://github.com/Rayrtfr/FasterTransformer.git
cd FasterTransformer
git submodule init && git submodule update
cd ..
mkdir triton_repo
mkdir -p triton_repo/models triton_repo/triton_config

# 编译
export TRITON_VERSION=22.12
cd ft_workspace
docker build \
    --build-arg TRITON_VERSION=${TRITON_VERSION} \
    -t triton_ft_backend:${TRITON_VERSION} \
    -f Dockerfile .

    # --build-arg http_proxy="http://172.19.0.1:7890" \
    # --build-arg https_proxy="http://172.19.0.1:7890" \


# docker run --rm -it \
#     --gpus="device=2" \
#     -p 9000:8000 -p 9001:8001 -p 9002:8002 \
#     -v $(pwd):/workspace  \
#     --name triton_ft_backend_pure \
#     triton_ft:${TRITON_VERSION} bash

# cd /workspace/fastertransformer_backend/
# mkdir build & cd build
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# export no_proxy=127.0.0.1,localhost
# cmake  -D CMAKE_EXPORT_COMPILE_COMMANDS=1 -D CMAKE_BUILD_TYPE=Release -D ENABLE_FP8=OFF ..
# make -j"$(grep -c ^processor /proc/cpuinfo)" install

# cd FasterTransformer
# mkdir build && cd build
# git submodule init && git submodule update
# pip3 install fire jax jaxlib transformers
# cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -D PYTHON_PATH=/usr/bin/python3 ..
# make -j12
# make install

# 转格式
cd ft_workspace
python huggingface_llama_convert.py  \
    -saved_dir=./triton_repo/models/llama \
    -in_file=../../FlagAlpha--Llama2-Chinese-7b-Chat/ \
    -infer_gpu_num=1 \
    -weight_data_type=fp16 \
    -model_name=llama

# 搞配置
config.pbtxt 
- tensor_para_size = gpu num
- model_checkpoint_path = /workspace/models/llama/1-gpu/ (注意这里是容器里的路径)

# 启动
cd ft_workspace
docker run --rm -it \
    --gpus="device=2" \
    --shm-size=4G \
    --name triton_ft_backend_pure \
    -v $(pwd)/triton_repo:/workspace \
    -p 9000:8000 -p 9001:8001 -p 9002:8002 \
    triton_ft_backend:${TRITON_VERSION} \
    tritonserver --model-repository=/workspace/triton_config/llama
```


## Reference

- [Release Notes :: NVIDIA Deep Learning Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-22-12.html#rel-22-12)
- [Llama2-Chinese/inference-speed/GPU/FasterTransformer_example at main · FlagAlpha/Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese/tree/main/inference-speed/GPU/FasterTransformer_example)