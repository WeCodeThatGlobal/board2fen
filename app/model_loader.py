from app.chessvision.core import ChessVision
from pathlib import Path

model = None

def load_model():
    global model
    if model is None:
        model = ChessVision(
            board_extractor_weights=str(Path("weights/best_yolo_extractor.pt")),
            classifier_weights=str(Path("weights/best_yolo_classifier.pt")),
            board_extractor_model_id="yolo",
            classifier_model_id="yolo",
            lazy_load=False
        )
    return model
