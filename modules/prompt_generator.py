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
            ]
        )
        generated_prompt = response['message']['content']
        generated_prompt = re.sub('\n', '', generated_prompt)

        return generated_prompt
