from hackaikit.core.base_module import BaseModule
from hackaikit.core.config_manager import ConfigManager
from hackaikit.integrations.huggingface_utils import get_hf_pipeline
from hackaikit.integrations.gemini_utils import GeminiUtil
from hackaikit.integrations.openai_utils import OpenAIUtil

class NLPModule(BaseModule):
    """
    Module for Natural Language Processing tasks.
    Supports Hugging Face Transformers, Gemini, and OpenAI.
    """
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.hf_sentiment_pipeline = None
        self.hf_text_generation_pipeline = None
        self.gemini_client = None
        self.openai_client = None

        if self.config_manager:
            # Initialize Hugging Face pipelines (example)
            default_sentiment_model = self.config_manager.get_setting('SETTINGS', 'DEFAULT_NLP_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english')
            self.hf_sentiment_pipeline = get_hf_pipeline("sentiment-analysis", model_name=default_sentiment_model, token=self.config_manager.get_huggingface_token())
            # self.hf_text_generation_pipeline = get_hf_pipeline("text-generation", model_name="gpt2", token=self.config_manager.get_huggingface_token()) # Example

            # Initialize Gemini
            gemini_key = self.config_manager.get_gemini_key()
            if gemini_key:
                self.gemini_client = GeminiUtil(api_key=gemini_key)
            else:
                print("NLPModule: Gemini API key not found. Gemini features will be unavailable.")

            # Initialize OpenAI
            openai_key = self.config_manager.get_openai_key()
            if openai_key:
                self.openai_client = OpenAIUtil(api_key=openai_key)
            else:
                print("NLPModule: OpenAI API key not found. OpenAI features will be unavailable.")
        else:
            print("NLPModule: ConfigManager not provided. API integrations will be limited.")


    def process(self, data, task="sentiment", **kwargs):
        if task == "sentiment":
            return self.sentiment_analysis(data, provider=kwargs.get("provider", "huggingface"))
        elif task == "generate_text":
            return self.generate_text(data, provider=kwargs.get("provider", "openai"), **kwargs)
        elif task == "summarize":
            return self.summarize_text(data, provider=kwargs.get("provider", "huggingface"), **kwargs) # Placeholder
        # Add more tasks like translation, Youtubeing etc.
        else:
            return "NLP task not supported."

    def sentiment_analysis(self, text, provider="huggingface"):
        if provider == "huggingface":
            if self.hf_sentiment_pipeline:
                return self.hf_sentiment_pipeline(text)
            return "Hugging Face sentiment pipeline not initialized."
        # Add sentiment analysis via Gemini/OpenAI if desired (though less direct)
        return "Provider not supported for sentiment analysis or not initialized."

    def generate_text(self, prompt, provider="openai", **kwargs):
        if provider == "openai" and self.openai_client:
            return self.openai_client.generate_chat_completion([{"role": "user", "content": prompt}], model=kwargs.get("model", "gpt-3.5-turbo"))
        elif provider == "gemini" and self.gemini_client:
            return self.gemini_client.generate_text(prompt)
        elif provider == "huggingface":
            if not self.hf_text_generation_pipeline: # Initialize on demand if not already
                 self.hf_text_generation_pipeline = get_hf_pipeline("text-generation", model_name=kwargs.get("model","gpt2"), token=self.config_manager.get_huggingface_token() if self.config_manager else None)
            if self.hf_text_generation_pipeline:
                return self.hf_text_generation_pipeline(prompt, max_length=kwargs.get("max_length", 50))
            return "Hugging Face text generation pipeline not initialized/failed."
        return f"Text generation provider '{provider}' not supported or not initialized."

    def summarize_text(self, text, provider="huggingface", **kwargs):
        # Example using Hugging Face summarization pipeline
        if provider == "huggingface":
            summarizer = get_hf_pipeline("summarization", model_name=kwargs.get("model", "facebook/bart-large-cnn"), token=self.config_manager.get_huggingface_token() if self.config_manager else None)
            if summarizer:
                return summarizer(text, max_length=kwargs.get("max_length", 150), min_length=kwargs.get("min_length",30), do_sample=False)
            return "Hugging Face summarization pipeline not initialized."
        # Placeholder for Gemini/OpenAI summarization (would involve specific prompting)
        elif provider == "openai" and self.openai_client:
            prompt = f"Summarize the following text:\n\n{text}"
            return self.openai_client.generate_chat_completion([{"role": "user", "content": prompt}], model=kwargs.get("model", "gpt-3.5-turbo"))
        return f"Summarization provider '{provider}' not supported or not initialized."


# Example Usage in a script or notebook
# if __name__ == '__main__':
#     # Create a .env file or config.ini with your API keys first
#     # Example .env file content:
#     # OPENAI_API_KEY=sk-yourkeyhere
#     # GEMINI_API_KEY=yourgeminikeyhere
#
#     # For this test, create a dummy config.ini if .env is not used
#     # with open("config.ini", "w") as f:
#     #     f.write("[API_KEYS]\n")
#     #     f.write("OPENAI_API_KEY = YOUR_OPENAI_KEY_OR_LEAVE_BLANK\n") # Replace or use .env
#     #     f.write("GEMINI_API_KEY = YOUR_GEMINI_KEY_OR_LEAVE_BLANK\n")   # Replace or use .env
#     #     f.write("[SETTINGS]\n")
#     #     f.write("DEFAULT_NLP_MODEL = distilbert-base-uncased-finetuned-sst-2-english\n")

#     config = ConfigManager(config_file='config.ini') # Will also try to load from .env
#     nlp_kit = NLPModule(config_manager=config)

#     # Hugging Face Sentiment
#     hf_sentiment = nlp_kit.sentiment_analysis("HackAI-Kit is awesome!", provider="huggingface")
#     print("HF Sentiment:", hf_sentiment)

#     # OpenAI Text Generation
#     if config.get_openai_key():
#         openai_text = nlp_kit.generate_text("Explain the concept of a Large Language Model in one sentence.", provider="openai")
#         print("OpenAI Generation:", openai_text)
#     else:
#         print("Skipping OpenAI generation test as API key is not configured.")

#     # Gemini Text Generation
#     if config.get_gemini_key():
#         gemini_text = nlp_kit.generate_text("What are the key features of the Gemini model?", provider="gemini")
#         print("Gemini Generation:", gemini_text)
#     else:
#         print("Skipping Gemini generation test as API key is not configured.")

#     # Hugging Face Text Generation
#     hf_generated_text = nlp_kit.generate_text("Once upon a time in a hackathon,", provider="huggingface", model="gpt2", max_length=30)
#     print("HF Generated Text:", hf_generated_text)

#     # Summarization
#     long_text = ("Large Language Models (LLMs) are advanced artificial intelligence systems "
#                  "designed to understand, generate, and manipulate human language. They are built "
#                  "using deep learning techniques, typically involving transformer architectures, "
#                  "and are trained on vast amounts of text data. This extensive training allows "
#                  "them to perform a wide range of language tasks, such as translation, summarization, "
#                  "question answering, and content creation, often with remarkable fluency and coherence. "
#                  "LLMs like GPT, BERT, and T5 have revolutionized various fields by enabling more "
#                  "natural and effective human-computer interactions.")
#     hf_summary = nlp_kit.summarize_text(long_text, provider="huggingface")
#     print("HF Summary:", hf_summary)
#     if config.get_openai_key():
#        openai_summary = nlp_kit.summarize_text(long_text, provider="openai")
#        print("OpenAI Summary:", openai_summary)
#     else:
#        print("Skipping OpenAI summarization test as API key is not configured.")