import tempfile
from fastapi import UploadFile
from app.model_loader import load_model

async def predict_fen(file: UploadFile) -> str:
    model = load_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    fen = model.image_to_fen(tmp_path)
    return fen
