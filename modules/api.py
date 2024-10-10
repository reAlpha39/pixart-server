import os
import asyncio
from modules.generate_image_request_model import ImageRequest
from modules.image_generator_2 import ImageGenerator, IMAGE_DIR

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse


image_generator = ImageGenerator()
app = FastAPI()

# Create a semaphore to limit concurrency for generate_image
generate_image_semaphore = asyncio.Semaphore(1)


@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    async with generate_image_semaphore:
        try:
            result = await asyncio.to_thread(image_generator.generate, request=request)
            return result
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{image_filename}")
def get_image(image_filename: str):
    image_path = os.path.join(IMAGE_DIR, image_filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.get("/")
def read_root():
    return {"message": "Welcome to the PixArt Sigma Image Generation API"}
