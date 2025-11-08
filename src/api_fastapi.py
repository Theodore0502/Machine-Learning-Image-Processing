from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI(title="Rice Leaf Health API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer/{task}")
async def infer(task: str, files: List[UploadFile] = File(...)):
    names = [f.filename for f in files]
    # TODO: run batch inference
    return {"task": task, "files": names, "results": "TODO"}
