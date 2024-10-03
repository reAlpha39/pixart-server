import requests


class PromptGenerator:
    @staticmethod
    def generate(prompt: str):
        # Call the Ollama API using the container's hostname
        url = "http://ollama:11434/api/chat"
        payload = {
            "model": "gemma2:latest",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "options": {
                "temperature": 0.2
            }
        }
        headers = {'Content-Type': 'application/json'}

        # Send a POST request to the Ollama container
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        print(response)

        if 'message' in response_data and 'content' in response_data['message']:
            text = response_data['message']['content'].strip()
            return text
        else:
            raise ValueError("Invalid response from Ollama API")
