# LLM课程使用指南


## 安装环境

step 1: 进入一个 terminal; （见视频)
输入： `conda create -p open-mmlab python=3.9 -y`

step 2: 激活环境 `conda activate /output/open-mmlab`

step 3: 安装相关的 package.  `sh install.sh`
这个 bash 安装本质上是执行（最重要的是：torch 和 cuda 版本要安装对）：
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U git+https://github.com/huggingface/trl
pip install -U transformers accelerate peft
pip install -U sentencepiece
pip install -U datasets bitsandbytes einops wandb
pip install scipy
pip install Jinja2
```


## 微调训练
> 推荐使用 trl 方式跑 SFT。上课讲得代码在：train_optimize/

step 1: 进入 trl_version_sft
`cd /openbayes/home/llm/llm_course/trl_version_sft`

step 2: 修改model path,;  （这个是故意留给大家自己去修改的）
点开 `v1.py` 和 `v2.py`

把下面这一行修改：
修改前：`model_path_or_name = "../../FlagAlpha--Llama2-Chinese-7b-Chat/"`
修改后：`model_path_or_name = "/openbayes/input/input0"`

> model_path_or_name 被挂在到了 /openbayes/input/input0， 实际上就是 FlagAlpha--Llama2-Chinese-7b-Chat 的别名

step 3: `sh run.sh`
> 记得查看各个参数的效果



### 小任务
1. 怎么定义自己的 metrics ??? （有的时候，我们不是只想看 loss, 有可能看其他的指标，比如 acc)
2. 了解 DataCollatorForCompletionOnlyLM 类，知道 什么叫做 mask prompt 的 loss ?
3. 改用 accelerate/deepspeed 跑通（deepspeed config 已经配置）, 两者其实差不多
> deepspeed 参考 train_optimize内的配置
> accelerate 参考：trl_version_sft

命令为：(可以放到 某个 sh 脚本上)
```
accelerate launch --config_file ./accelerate_config.yaml v1.py --output_dir "./output/" 还有很多参数，（按照 run.sh 填上）
```




## Gradio Demo

1. 检查配置：模型路径
2. 启动服务
3. 交互

```bash
python demo_gradio.py 
```

## 量化


### GGML/GGUF

1. 下载源代码并编译
2. 量化：先到fp16，再到int
3. 使用量化后模型推理

```bash
# 下载编译
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
# 量化
python convert.py ../FlagAlpha--Llama2-Chinese-7b-Chat/ /shared/ggml-model-f16.gguf
./quantize /shared/ggml-model-f16.gguf  /shared/ggml-model-q4_0.gguf q4_0
# 推理
./main \
    -m /shared/ggml-model-q4_0.gguf \
    -p "<s>Human: 介绍一下中国\n</s><s>Assistant: "
```

### AutoGPTQ

1. 安装library
2. 量化
3. 使用量化模型推理

```bash
# 注意检查路径配置
python gptq.py
python gptq_gen.py
```


## TritonServer/SSE

1. 检查配置：Tokenizer路径、host地址
2. 启动后端SSE接口服务
3. 前端验证

```bash
pip install uvicorn
uvicorn main:app --host 0.0.0.0 --port 8080

```