"""FastAPI app for the Report 4 demo.

Run from the repo root with ``demo/run_api.ps1`` (sets JAVA_HOME for VnCoreNLP), or:
    python -m uvicorn demo.api.app:app --host 127.0.0.1 --port 8000 --workers 1

Frontend on the same port (no CORS, one command):
- the page ``demo/web/index.html`` (or ``DEMO_WEB_FILE``) is served at ``/`` and an ``assets/`` folder
  next to it (fonts, images) is served at ``/assets``.
- otherwise a built frontend at ``demo/web/dist`` is served at ``/``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import SAMPLES, DemoService, InputError

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
WEB_FILE = (
    Path(os.environ["DEMO_WEB_FILE"])
    if os.environ.get("DEMO_WEB_FILE")
    else Path(__file__).resolve().parents[1] / "web" / "index.html"
)
_state: dict[str, DemoService] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["service"] = DemoService()
    yield
    _state.clear()


app = FastAPI(title="Sentiment Lens API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Câu tiếng Việt thô, chưa tách từ.")
    model: str = Field("uit", description="'uit' hoặc 'vlsp'.")
    lexicon: bool = Field(True, description="False: tắt nhánh lexicon lúc suy luận (không phải baseline A).")


def _service() -> DemoService:
    return _state["service"]


@app.exception_handler(InputError)
async def _input_error(_request: Request, error: InputError):
    return JSONResponse(status_code=400, content={"error": str(error)})


@app.exception_handler(RequestValidationError)
async def _invalid_body(_request: Request, _error: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": 'Dữ liệu gửi lên không đúng định dạng. Cần JSON {"text": "...", "model": "uit", "lexicon": true}.'},
    )


@app.exception_handler(StarletteHTTPException)
async def _http_error(_request: Request, error: StarletteHTTPException):
    return JSONResponse(status_code=error.status_code, content={"error": str(error.detail)})


@app.exception_handler(FileNotFoundError)
async def _missing_asset(_request: Request, error: FileNotFoundError):
    return JSONResponse(status_code=503, content={"error": str(error)})


@app.get("/api/health")
def health() -> dict:
    service = _service()
    return {"status": "ok", "device": service.device, "loaded": service.loaded()}


@app.get("/api/models")
def models() -> list[dict]:
    return _service().models()


@app.get("/api/samples")
def samples() -> list[str]:
    return list(SAMPLES)


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict:
    return _service().predict(request.text, request.model, request.lexicon)


if WEB_FILE is not None and WEB_FILE.is_file():

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_FILE, media_type="text/html; charset=utf-8")

    # Only the page and an optional assets/ folder next to it (fonts, images) are exposed, never the
    # whole folder: with run_api.ps1 -Lan anyone on the network can reach this server.
    if (WEB_FILE.parent / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_FILE.parent / "assets"), name="web-assets")
elif WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
