from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image 
import io, os, json
from fastapi import HTTPException
from app.pipeline.analyzer import AnalysePlantImage

app = FastAPI(title="Plant Intelligence API", description="AI-powered plant disease detection and analysis")

@app.get("/health")
def health():
    return {"Status": "Alive",
            "Confidence": "100%"}
@app.post("/api/v1/plant-intelligence/analyze")


async def analyse(
    image: UploadFile = File(...), 
    crop: str = Form(default=None),
    session_id: str | None = Form(default=None),
):
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Only JPEG/PNG/jpg images are supported, try again with a valid format")
    if crop not in ["Tomato", "Potato", "Other", "vine", "olive"]:
        raise HTTPException(status_code=400, detail="Invalid crop type, try again with a valid crop type")
    
    raw = await image.read()

    # Reject huge files
    if len(raw) > 16 * 1024 * 1024:  # 16 MB
        raise HTTPException(status_code=413, detail="Image too large (max 16MB)")
    
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    result = AnalysePlantImage(img, user_crop=crop, session_id=session_id)
    session_id = result.session_id
    session_folder= os.path.join("sessions", session_id)
    os.makedirs(session_folder, exist_ok=True)
    image_path = os.path.join(session_folder, "input_image.jpg")
    with open(image_path, "wb") as f:
        f.write(raw)
    
    result_path = os.path.join(session_folder, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    return result.model_dump()
    

