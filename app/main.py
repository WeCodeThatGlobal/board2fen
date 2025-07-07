from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.predictor import predict_fen

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-fen")
async def predict_fen_endpoint(file: UploadFile = File(...)):
    try:
        fen = await predict_fen(file)
        return {"fen": fen}
    except Exception as e:
        return {"error": str(e)}
