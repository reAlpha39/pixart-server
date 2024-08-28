from pydantic import BaseModel
from typing import Optional

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "low quality, bad, watermark, human, people, person"
    seed: Optional[int] = 42
    randomize_seed: Optional[bool] = False
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    guidance_scale: Optional[float] = 3.5
    num_inference_steps: Optional[int] = 28
    url_output_type: Optional[bool] = False
    generate_prompt: Optional[bool] = False
