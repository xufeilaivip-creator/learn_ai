# llm

llm课程代码


```bash

# 环境
cd /openbayes/home/llm_course
python -m venv .venv
source .venv/bin/activate
pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 查看模型和数据

ll /input0
ll /input1

####### Ch5 #######

# 训练
bash llm_course.sh


####### Ch6 #######

# Gradio
python demo_gradio.py 

# 量化
python gptq.py
python gptq_gen.py
cd llama.cpp
make
python convert.py ../../FlagAlpha--Llama2-Chinese-7b-Chat
cd ..
mkdir ggml_output
mv ../FlagAlpha--Llama2-Chinese-7b-Chat/ggml-model-f16.gguf  ./ggml_output/
./llama.cpp/quantize ./ggml_output/ggml-model-f16.gguf  ./ggml_output/ggml-model-q4_0.gguf q4_0 
./llama.cpp/main \
    -m ggml_output/ggml-model-q4_0.gguf \
    -p "<s>Human: 介绍一下中国\n</s><s>Assistant: "

# TritonServer&SSE
cd llm_course/triton_server/ft_workspace
python huggingface_llama_convert.py  \
    -saved_dir=./triton_repo/models/llama \
    -in_file=../../FlagAlpha--Llama2-Chinese-7b-Chat/ \
    -infer_gpu_num=1 \
    -weight_data_type=fp16 \
    -model_name=llama

# 换国内源
cat > /etc/apt/sources.list << END  
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
END
apt update
apt  install docker.io
docker pull yamhouston/triton_ft_backend:22.12

docker run --rm -it \
    --gpus="device=0" \
    --shm-size=4G \
    --name triton_ft_backend_pure \
    -v $(pwd)/triton_repo:/workspace \
    -p 9000:8000 -p 9001:8001 -p 9002:8002 \
    yamhouston/triton_ft_backend:22.12 \
    tritonserver --model-repository=/workspace/triton_config/llama

cd ..
uvicorn main:app --host 0.0.0.0 --port 8080

```

