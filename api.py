from fastapi import FastAPI, Body, File, UploadedFile, status, HTTPException
import base64
from fastapi.responses import JSONResponse  



app = FastAPI(
    title="VisionIQ"
)


@app.get("/")
async def root():
    return {"message":"Welcome to VisionIQ!!"}


@app.get("/health")
def health():
    return {"message":"OK"}


@app.post("")