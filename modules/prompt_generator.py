import ollama
import re

class PromptGenerator:
    def generate(prompt: str):
        response = ollama.chat(
            model='gemma2:latest',
            keep_alive=0,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                "temperature": 0
            }
        )
        text = response['message']['content'].strip()
        return text
