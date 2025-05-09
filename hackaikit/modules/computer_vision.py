from hackaikit.core.base_module import BaseModule
from hackaikit.core.config_manager import ConfigManager
from hackaikit.integrations.huggingface_utils import get_hf_pipeline, load_hf_object_detection_model
from hackaikit.integrations.gemini_utils import GeminiUtil
import cv2
from PIL import Image # For working with Hugging Face models and Gemini

class ComputerVisionModule(BaseModule):
    """
    Module for Computer Vision tasks.
    Integrates OpenCV, Hugging Face Vision models, and Gemini Vision.
    """
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.hf_object_detection_pipeline = None
        self.hf_image_classification_pipeline = None
        self.yolo_model = None # Placeholder for local YOLO model
        self.gemini_client = None

        if self.config_manager:
            # Initialize Gemini for Vision
            gemini_key = self.config_manager.get_gemini_key()
            if gemini_key:
                self.gemini_client = GeminiUtil(api_key=gemini_key)
            else:
                print("ComputerVisionModule: Gemini API key not found. Gemini Vision features will be unavailable.")

            # Example: Load a default HF image classification pipeline
            self.hf_image_classification_pipeline = get_hf_pipeline(
                task="image-classification",
                model_name=self.config_manager.get_setting('SETTINGS', 'DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL', 'google/vit-base-patch16-224'),
                token=self.config_manager.get_huggingface_token()
            )
            # Object detection can be more complex to set up a default pipeline directly for all models
            # It's often better to load model and feature_extractor separately for HF object detection
            # For YOLO, you'd typically load it from torch.hub or local files.

    def process(self, data, task="object_detection", **kwargs):
        """
        Data is typically an image path or a pre-loaded image (e.g., NumPy array from OpenCV or PIL Image).
        """
        if not isinstance(data, (str, Image.Image)): # For HF and Gemini, PIL Image is good
             # If it's an OpenCV image (numpy array), convert to PIL
            if hasattr(data, 'shape'): # Basic check for numpy array
                try:
                    data = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    return f"Error converting OpenCV image to PIL Image: {e}"
            else:
                return "CVModule: Invalid data type for image. Expecting path, PIL Image, or OpenCV image."


        if task == "object_detection":
            return self.object_detection(data, provider=kwargs.get("provider", "huggingface"), **kwargs)
        elif task == "image_classification":
            return self.image_classification(data, provider=kwargs.get("provider", "huggingface"), **kwargs)
        elif task == "image_captioning_gemini": # Specific Gemini task
             if not self.gemini_client:
                return "Gemini client not initialized. Please check API key."
             prompt = kwargs.get("prompt", "Describe this image in detail.")
             return self.gemini_client.analyze_image(prompt, data) # data here should be path or PIL.Image
        # Add more tasks: segmentation, ocr, etc.
        else:
            return "CV task not supported."

    def object_detection(self, image_input, provider="huggingface", **kwargs):
        """image_input should be a PIL Image or path for HF/Gemini."""
        if provider == "huggingface":
            model_name = kwargs.get("model", self.config_manager.get_setting('SETTINGS', 'DEFAULT_CV_OBJECT_DETECTION_MODEL', "facebook/detr-resnet-50")) # Example DETR model
            if not self.hf_object_detection_pipeline or self.hf_object_detection_pipeline.model.name_or_path != model_name:
                 self.hf_object_detection_pipeline = get_hf_pipeline("object-detection", model_name=model_name, token=self.config_manager.get_huggingface_token() if self.config_manager else None)

            if self.hf_object_detection_pipeline:
                # Ensure image_input is in a format the pipeline expects (often PIL Image or path)
                pil_image = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
                return self.hf_object_detection_pipeline(pil_image)
            return "Hugging Face object detection pipeline not initialized."

        elif provider == "yolo": # Basic placeholder for YOLO
            # For YOLO, you'd typically load a pre-trained model (e.g., from PyTorch Hub)
            # and then pass the image (often as a NumPy array) to it.
            # Example:
            # if self.yolo_model is None:
            #   import torch
            #   try:
            #       self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            #   except Exception as e:
            #       return f"Failed to load YOLOv5 model: {e}"
            # if self.yolo_model:
            #   # YOLOv5 expects image path, PIL image, or numpy array
            #   results = self.yolo_model(image_input)
            #   return results.pandas().xyxy[0].to_dict(orient="records") # Example output
            return "YOLO integration placeholder: Implement model loading and inference."
        
        elif provider == "gemini_vision_ocr" and self.gemini_client: # Example for OCR like task
            prompt = "Extract all text from this image."
            return self.gemini_client.analyze_image(prompt, image_input)

        return f"Object detection provider '{provider}' not supported or not initialized."

    def image_classification(self, image_input, provider="huggingface", **kwargs):
        """image_input should be a PIL Image or path for HF."""
        if provider == "huggingface":
            if self.hf_image_classification_pipeline:
                 pil_image = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
                 return self.hf_image_classification_pipeline(pil_image)
            return "Hugging Face image classification pipeline not initialized."
        return f"Image classification provider '{provider}' not supported or not initialized."

# Example Usage:
# if __name__ == '__main__':
#     # Ensure you have an image file, e.g., 'test_image.jpg'
#     # from PIL import Image, ImageDraw
#     # img = Image.new('RGB', (400, 300), color = (73, 109, 137))
#     # d = ImageDraw.Draw(img)
#     # d.text((10,10), "Hello World", fill=(255,255,0))
#     # img.save("test_image.jpg")

#     config = ConfigManager(config_file='config.ini') # Assumes config.ini or .env is set up
#     cv_kit = ComputerVisionModule(config_manager=config)

#     image_path = "test_image.jpg" # Replace with your image path

#     if not os.path.exists(image_path):
#         print(f"Test image {image_path} not found. Please create it.")
#     else:
#         # Hugging Face Image Classification
#         hf_classification_results = cv_kit.image_classification(image_path, provider="huggingface")
#         print("HF Image Classification:", hf_classification_results)

#         # Hugging Face Object Detection
#         # Note: Default object detection models can be large and might take time to download.
#         # Some models like DETR are better suited for this pipeline approach.
#         # Others (like some YOLOs on HF) might need more specific handling of feature extractors.
#         hf_od_results = cv_kit.object_detection(image_path, provider="huggingface", model="facebook/detr-resnet-50") # Using a common HF OD model
#         print("HF Object Detection:", hf_od_results)


#         # Gemini Vision (if API key is configured)
#         if config.get_gemini_key():
#             gemini_caption = cv_kit.process(image_path, task="image_captioning_gemini", prompt="What objects are in this image?")
#             print("Gemini Vision Caption:", gemini_caption)
#         else:
#             print("Skipping Gemini Vision test as API key is not configured.")