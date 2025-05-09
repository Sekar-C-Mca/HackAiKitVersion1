import openai

class OpenAIUtil:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)


    def generate_chat_completion(self, messages, model="gpt-3.5-turbo"):
        """
        Generates a chat completion using OpenAI's API.
        messages: A list of message objects, e.g., [{"role": "user", "content": "Hello!"}]
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating chat completion with OpenAI: {e}"

    def generate_text(self, prompt, model="gpt-3.5-turbo-instruct", max_tokens=150): # Older completion endpoint model
        """Generates text using OpenAI's completion API (legacy, prefer chat)."""
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response.choices[0].text.strip()
        except Exception as e:
            # Fallback for models that might not be available or if using chat based model name
             if "gpt-3.5-turbo" in model or "gpt-4" in model: # If it's a chat model, use chat completion
                return self.generate_chat_completion([{"role": "user", "content": prompt}], model=model)
             return f"Error generating text with OpenAI: {e}"


# Example usage:
# if __name__ == '__main__':
#     from hackaikit.core.config_manager import ConfigManager
#     config = ConfigManager()
#     openai_api_key = config.get_openai_key()
#     if openai_api_key:
#         openai_client = OpenAIUtil(api_key=openai_api_key)
#         chat_response = openai_client.generate_chat_completion(
#             messages=[{"role": "user", "content": "What is the capital of France?"}]
#         )
#         print("OpenAI Chat Response:", chat_response)
#     else:
#         print("OpenAI API key not configured. Skipping OpenAIUtil tests.")