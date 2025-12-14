import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # noqa: I001
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch.nn import functional
from transformers import CLIPTokenizer
from typing_extensions import TypedDict
import faiss

from clip_search.module import VideoSearchLightningModule


class Resources(TypedDict):
    model: VideoSearchLightningModule
    tokenizer: CLIPTokenizer
    indexer: faiss.IndexFlatIP
    mapping: dict[int, str]
    device: str


class SearchRequest(BaseModel):
    query: str
    k_search: int


class SearchResponse(BaseModel):
    videos: list[str]
    distances: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    with open("./paths.json") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoSearchLightningModule.load_from_checkpoint("best.ckpt")
    model.eval()
    model.to(device)

    tokenizer = CLIPTokenizer.from_pretrained(model.model_name)
    indexer = faiss.read_index("vector.index")

    app.state.resources = {
        "model": model,
        "tokenizer": tokenizer,
        "indexer": indexer,
        "mapping": data,
        "device": device,
    }

    yield

    del app.state.resources


app = FastAPI(lifespan=lifespan)
app.mount("/data", StaticFiles(directory="data"), name="data")


@app.post("/search")
def search(request: Request, search_request: SearchRequest) -> SearchResponse:
    resources = app.state.resources
    device = resources["device"]

    inputs = resources["tokenizer"](
        search_request.query,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )

    with torch.inference_mode():
        inputs = {k: v.to(device) for k, v in inputs.items()}

        text_features = resources["model"].model.clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        normalized_query = functional.normalize(text_features, dim=1)

        query_vector = normalized_query.detach().cpu().numpy().astype("float32")

    distances, ids = resources["indexer"].search(query_vector, search_request.k_search)

    found_paths = []
    for idx in ids[0]:
        if idx == -1:
            continue
        found_paths.append(resources["mapping"][str(idx)])

    return SearchResponse(
        videos=found_paths,
        distances=distances[0].tolist(),
    )


if __name__ == "__main__":
    uvicorn.run(app, workers=1)
