python convert.py models/llama2-7B/
./quantize \
    ./models/llama2-7B/ggml-model-f16.gguf  \
    ./models/llama2-7B/ggml-model-q4_0.gguf q4_0
./main \
    -m ggml_output/ggml-model-q4_0.gguf \
    -p "<s>Human: 介绍一下中国\n</s><s>Assistant: "

python convert.py models/llama2-7B/  --vocab-only --outfile models/llama2-7B/ggml-vocab-llama.gguf
