import google.generativeai as genai

class GeminiUtil:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro') # For text
        self.vision_model = genai.GenerativeModel('gemini-pro-vision') # For multimodal

    def generate_text(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating text with Gemini: {e}"

    def analyze_image(self, prompt, image_path_or_bytes):
        """
        Analyzes an image with a prompt.
        image_path_or_bytes: Can be a path to an image file or image bytes.
        """
        from PIL import Image
        import io

        try:
            if isinstance(image_path_or_bytes, str): # Path
                img = Image.open(image_path_or_bytes)
            elif isinstance(image_path_or_bytes, bytes): # Bytes
                img = Image.open(io.BytesIO(image_path_or_bytes))
            else: # PIL Image object
                img = image_path_or_bytes
            
            response = self.vision_model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            return f"Error analyzing image with Gemini Vision: {e}"

    def multimodal_interaction(self, contents):
        """
        Allows for more complex multimodal interactions.
        contents: A list of parts (text, image).
                  e.g., ["Describe this image:", PIL.Image.open('image.jpg')]
        """
        try:
            response = self.vision_model.generate_content(contents)
            return response.text
        except Exception as e:
            return f"Error with Gemini multimodal interaction: {e}"

# Example usage (would be in a module or script):
# if __name__ == '__main__':
#     from hackaikit.core.config_manager import ConfigManager
#     config = ConfigManager()
#     gemini_api_key = config.get_gemini_key()
#     if gemini_api_key:
#         gemini_client = GeminiUtil(api_key=gemini_api_key)
#         text_response = gemini_client.generate_text("Explain quantum computing in simple terms.")
#         print("Gemini Text Response:", text_response)
#
#         # Create a dummy image file for testing
#         from PIL import Image
#         try:
#             dummy_image = Image.new('RGB', (60, 30), color = 'red')
#             dummy_image.save("dummy_image.png")
#             image_response = gemini_client.analyze_image("What is in this image?", "dummy_image.png")
#             print("Gemini Image Response:", image_response)
#         except Exception as e:
#             print(f"Skipping Gemini image test due to an error: {e}")
#     else:
#         print("Gemini API key not configured. Skipping GeminiUtil tests.")