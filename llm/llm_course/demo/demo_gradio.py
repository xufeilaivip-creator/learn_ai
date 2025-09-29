import argparse
from typing import List, Tuple
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Chatter:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.max_len = self.model.config.max_length
    
    def build_prompt(self, query: str, history, max_new_tokens: int) -> str:
        """
        [("你好", "你也好"), ..]
        """
        prompt_max_len = self.max_len - max_new_tokens
        last_inp = f"""<s>Human: {query}
</s><s>Assistant: """
        new_inp = last_inp
        for (q, r) in history[::-1]:
            curr_inp = f"""<s>Human: {q}
</s><s>Assistant: {r}
</s>"""
            new_tmp_inp = curr_inp + last_inp
            ids = self.tokenizer.encode(new_tmp_inp)
            if len(ids) > prompt_max_len:
                new_inp = last_inp
                break
            else:
                new_inp = new_tmp_inp
            last_inp = new_inp
        return new_inp
    
    def chat(
        self,
        query: str,
        history: List[Tuple], 
        max_new_tokens: int, 
        top_p: float, 
        temperature: float,
    ) -> Tuple[str, Tuple[str, str]]:
        prompt = self.build_prompt(query, history, max_new_tokens)
        print("=====Query=====")
        print(query)
        print("=====History=====")
        print(history)
        print("=====Prompt=====")
        print(prompt)
        print("-*" * 30)
        print()
        inputs = self.tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        inputs.to(self.model.device)
        generate_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.3,
            "streamer": self.streamer,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        generate_input = {**inputs, **generate_config}
        thread = Thread(target=self.model.generate, kwargs=generate_input)
        thread.start()
        response = ""
        for new_text in self.streamer:
            new_text = new_text.strip("\n").strip(self.tokenizer.eos_token)
            response += new_text
            new_history = history + [(query, response)]
            yield response, new_history


def main():
    parser = argparse.ArgumentParser(description="Humanable Chat GXX Finetune")
    parser.add_argument(
        "--model", type=str, default=None, metavar="ID/PATH", required=False,
        help="[model] LLM model id or model path (default: None)"
    ) 
    args = parser.parse_args()
    # MODEL_PATH = "../FlagAlpha--Llama2-Chinese-7b-Chat/"
    MODEL_PATH = "/openbayes/input/input0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    model.eval();
    model.to("cuda:0")
    chatter = Chatter(model, tokenizer)
    
    
    def reset_user_input():
        return gr.update(value="")

    def reset_state():
        return [], []

    def chat(
        query: str, 
        chatbot: list, 
        max_new_tokens: int, 
        top_p: float, 
        temperature: float, 
        history: list
    ):
        chatbot.append((query, ""))
        for response, history in chatter.chat(
            query, history, max_new_tokens, top_p, temperature
        ):
            chatbot[-1] = (query, response)
            yield chatbot, history

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">MyOwn ChatBot</h1>""")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=1):
                    user_input = gr.Textbox(show_label=False, placeholder="Input", lines=4).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                empty_btn = gr.Button("Clear History")
                max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Maximum generated length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.1, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.2, step=0.1, label="Temperature", interactive=True)

        history = gr.State([])
        submit_btn.click(
            chat, 
            inputs=[user_input, chatbot, max_new_tokens, top_p, temperature, history], 
            outputs=[chatbot, history], 
            show_progress=True
        )
        submit_btn.click(reset_user_input, [], [user_input])
        empty_btn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    try:
        import requests
        meta = requests.get("http://localhost:21999/gear-status", timeout=5).json()
        url = meta["links"].get("auxiliary")
        if url:
            print(f"Open from: {url}")
    except Exception as e:
        pass
    demo.queue().launch(server_name="0.0.0.0", server_port=8080, share=False)


if __name__ == "__main__":
    main()