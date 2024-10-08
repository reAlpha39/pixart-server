import ollama
import re

class PromptGenerator:
    def generate(prompt: str, prompt_model: str):
        try:
            response = ollama.chat(
                model=prompt_model,
                keep_alive=keep_alive_prompt_model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
            )
            text = response['message']['content'].strip()
            return text

        except Exception as e:
            print(e)
            raise Exception(str(e))
