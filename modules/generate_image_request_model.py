from pydantic import BaseModel
from typing import Optional

class ImageRequest(BaseModel):
    prompt: str
    model: str = "PixArt-Sigma-900M"
    negative_prompt: str = "low quality, bad, watermark, human, people, person"
    seed: int = 42
    randomize_seed: bool = False
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    generate_prompt: bool = False
    generate_prompt_temperature: float = 0.8
    prompt_model: str = "gemma2:latest"
    keep_alive_prompt_model: int = 0
