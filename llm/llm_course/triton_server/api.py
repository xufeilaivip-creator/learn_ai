from typing import Literal, List, Generator, Dict
from dataclasses import asdict, dataclass
import asyncio
import json, uuid

import pnlp
from fastapi import Request

from llm_triton import LlamaLlm


@dataclass
class GenerateConfig:

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.7
    top_k: int = 30
    repetition_penalty: float = 1.0


@dataclass
class LlmMessage:

    request_id: str
    message_id: str
    content: str
    status: str


llm_config_dict = {
    "model_id": "//openbayes/home/llm/llm_course/FlagAlpha--Llama2-Chinese-7b-Chat/",
    "triton_host": "115.159.25.224:10111",
    "triton_model": "fastertransformer",
}


class LlmServer:

    def __init__(self):
        llm_config = pnlp.MagicDict(llm_config_dict)
        self.gen_config = GenerateConfig()
        self.llm_ins = LlamaLlm(
            model_id=llm_config.model_id,
            triton_host=llm_config.triton_host,
            triton_model=llm_config.triton_model,
        )

    def stream_generate(
        self,
        prompts: List[str],
    ) -> Generator[List[str], None, None]:
        # for resp_texts in model.generate(
        #     ...
        for resp_texts in self.llm_ins.generate(
            prompts,
            self.gen_config.max_new_tokens,
            self.gen_config.temperature,
            self.gen_config.top_p,
            self.gen_config.top_k,
            self.gen_config.repetition_penalty,
        ):
            yield resp_texts
    
    def build_event_msg(
        self, 
        request_id: str, 
        message_id: str,
        resp_str: str,
        status: str, 
    ) -> Dict:
        msg = LlmMessage(request_id, message_id, resp_str, status)
        dct = asdict(msg)
        out_msg = json.dumps(dct)
        return out_msg
    
    async def stream_run(
        self,
        request: Request,
        q: str,
    ) -> Generator[Dict, None, None]:
        rid = str(uuid.uuid4())        
        if await request.is_disconnected():
            yield
        
        texts = [q]
        i = 0
        sents = []
        for txt in texts:
            sents.append(f"Human: {txt} \n\nAssistant: ")

        for resp_text in self.stream_generate(sents):
            resp_str = resp_text[0].replace(sents[0], "")
            yield self.build_event_msg(rid, str(i), resp_str, "in_progress")
            await asyncio.sleep(0.1)
            i += 1
        
        yield self.build_event_msg(rid, str(i), "", "stop")


llm_server = LlmServer()