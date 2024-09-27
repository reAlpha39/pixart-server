import ollama
import re

class PromptGenerator:
    def generate(prompt: str):
        response = ollama.chat(
            model='gemma2:latest',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                "temperature": 0.2
            }
        )
        text = response['message']['content'].strip()
        return text
