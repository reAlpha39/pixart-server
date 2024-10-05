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
        )
        text = response['message']['content'].strip()
        return text
