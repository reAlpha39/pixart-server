from llama_cpp import Llama
import re

class PromptGenerator:
    def __init__(self):
        self.llm = Llama(
            model_path="./gemma-2-9b-it-Q4_K_M.gguf",
            n_gpu_layers=-1,
            echo=False,
            verbose=False,
            chat_format="llama-2",
        )

    def generate(self, prompt: str):
        response = self.llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": prompt,
            }]
        )

        text = response['choices'][0]['message']['content'].strip()
        return text
