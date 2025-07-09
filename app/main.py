from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.predictor import predict_fen

load_dotenv()

app = FastAPI()

origins = [
    "https://api.chessplay.io",
    "https://api.saaschess.com",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_TOKEN = os.getenv("FEN_API_SECRET")

def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    token = auth_header.split(" ")[1]
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

@app.post("/predict-fen")
async def predict_fen_endpoint(request: Request, file: UploadFile = File(...)):
    verify_token(request)

    try:
        fen = await predict_fen(file)
        return {"fen": fen}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")