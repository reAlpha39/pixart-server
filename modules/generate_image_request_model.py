from pydantic import BaseModel
from typing import Optional

class ImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "PixArt-Sigma-900M"
    negative_prompt: Optional[str] = "low quality, bad, watermark, human, people, person"
    seed: Optional[int] = 42
    randomize_seed: Optional[bool] = False
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    guidance_scale: Optional[float] = 3.5
    num_inference_steps: Optional[int] = 28
    generate_prompt: Optional[bool] = False,
    prompt_model: Optional[str] = "gemma2:latest",
    keep_alive_prompt_model: Optional[int] = 0,
    use_resolution_binning: Optional[bool] = True,
