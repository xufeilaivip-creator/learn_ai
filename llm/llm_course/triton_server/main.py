from fastapi import Request, FastAPI
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

from api import llm_server


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    import requests
    meta = requests.get("http://localhost:21999/gear-status", timeout=5).json()
    url = meta["links"].get("auxiliary")
    if url:
        print(f"Open from: {url}")
except Exception as e:
    pass


@app.get("/steam_demo")
async def demo_sse_svc(request: Request, q: str):
    import json
    import time
    def gen(req, q):
        for i in range(10):
            cont = q + str(i)
            msg = {
                "status": "inprocess",
                "content": cont,
            }
            yield json.dumps(msg)
            time.sleep(1)
        msg = {
            "status": "stop",
            "content": ""
        }
        yield json.dumps(msg)

    event_generator = gen(request, q)
    return EventSourceResponse(event_generator)


@app.get("/steam_ask")
async def ask_sse_svc(request: Request, q: str):
    event_generator = llm_server.stream_run(request, q)
    return EventSourceResponse(event_generator)