import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT, CORS_ORIGINS
from routes import router
import rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG pipeline on startup
    rag_pipeline.initialize()
    yield


app = FastAPI(title="MedQuery RAG API", version="1.0.0", lifespan=lifespan)

# CORS — allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
