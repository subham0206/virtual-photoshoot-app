import io
import base64
import requests
from PIL import Image
import backoff
from typing import Tuple, Dict, Optional, Any, List
import time
import json

class OpenAIDALLE3Handler:
    """
    Handler class for OpenAI's DALL-E 3 integration for better color and texture preservation.
    """
    def __init__(self, api_key: str, timeout: int = 60, model: str = "dall-e-3"):
        """
        Initialize the OpenAIDALLE3Handler with the OpenAI API key
        
        Args:
            api_key: The OpenAI API key
            timeout: Timeout in seconds for API calls (default: 60)
            model: The DALL-E model to use (default: dall-e-3)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        """
        buffered = io.BytesIO()
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, 
                          requests.exceptions.Timeout, 
                          requests.exceptions.ConnectionError),
                         max_tries=3)
    def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard", n: int = 1) -> Optional[Image.Image]:
        """
        Generate an image using DALL-E 3 API
        
        Args:
            prompt: Text description of the desired image
            size: Size of the generated image (1024x1024, 1024x1792, or 1792x1024)
            quality: Image quality (standard or hd)
            n: Number of images to generate
            
        Returns:
            PIL Image object if successful, None otherwise
        """
        try:
            url = f"{self.base_url}/images/generations"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "n": n,
                "size": size,
                "quality": quality,
                "response_format": "b64_json"
            }
            
            print(f"Sending request to DALL-E 3 API with prompt: {prompt[:100]}...")
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if "data" in response_data and len(response_data["data"]) > 0:
                    image_data = response_data["data"][0]
                    if "b64_json" in image_data:
                        image_bytes = base64.b64decode(image_data["b64_json"])
                        image = Image.open(io.BytesIO(image_bytes))
                        return image
            
            # Handle error responses
            error_msg = f"API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return None
            
        except Exception as e:
            print(f"Error generating image with DALL-E 3: {str(e)}")
            return None
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, 
                          requests.exceptions.Timeout, 
                          requests.exceptions.ConnectionError),
                         max_tries=3)
    def fit_apparel_on_model(self,
        apparel_image: Image.Image,
        model_attributes: Dict[str, str],
        pose: str,
        model_image: Image.Image = None,
        photoshoot_settings: Dict[str, Any] = None,
        styling_prompt: str = "",
        apparel_color: str = None
    ) -> Tuple[Optional[Image.Image], Optional[str], Optional[Image.Image]]:
        """
        Generate an image of apparel fitted on a model with the specified attributes using DALL-E 3
        
        Args:
            apparel_image: The uploaded apparel image
            model_attributes: Dictionary of model attributes
            pose: The desired pose
            model_image: Optional previously generated model image
            photoshoot_settings: Optional dictionary with photoshoot settings
            styling_prompt: Optional styling instructions for the apparel
            apparel_color: Optional color to change the apparel to (hex code)
            
        Returns:
            Tuple containing:
            - composite image (with apparel, model, and final result)
            - status message
            - standalone high quality fitted image (model wearing apparel only)
        """
        try:
            # Set default photoshoot settings if not provided
            if photoshoot_settings is None:
                photoshoot_settings = {
                    "view_angle": "Front view",
                    "background": "Studio white",
                    "lighting_style": "Standard studio",
                    "quality": "Standard"
                }
            
            # Extract view angle, background, and lighting settings
            view_angle = photoshoot_settings.get("view_angle", "Front view")
            background = photoshoot_settings.get("background", "Studio white")
            lighting_style = photoshoot_settings.get("lighting_style", "Standard studio")
            
            # Create model description based on attributes
            model_description_parts = []
            if 'gender' in model_attributes:
                model_description_parts.append(f"{model_attributes['gender']}")
            if 'ethnicity' in model_attributes:
                model_description_parts.append(model_attributes['ethnicity'])
            if 'skin_tone' in model_attributes:
                model_description_parts.append(f"with {model_attributes['skin_tone']}")
            if 'build' in model_attributes:
                model_description_parts.append(f"{model_attributes['build']} build")
            if 'hair_color' in model_attributes and 'hair_style' in model_attributes:
                model_description_parts.append(f"{model_attributes['hair_color']} {model_attributes['hair_style']} hair")
            if 'facial_hair' in model_attributes and model_attributes.get('facial_hair', "None") != "None":
                model_description_parts.append(f"with {model_attributes['facial_hair']}")
            
            # Join all parts to create the description
            model_description = ", ".join(model_description_parts)
            
            # Extract the dominant color of the apparel
            apparel_color_desc = apparel_color if apparel_color else "original color"
            
            # Create a comprehensive prompt for DALL-E 3
            base_prompt = f"""
            Create a high-quality photorealistic image of a {model_description} fashion model wearing the exact 
            apparel that I've uploaded. This is for a virtual try-on application where color and texture preservation is critical.
            
            MODEL DESCRIPTION:
            - {model_description}
            - The model should be in a {pose} pose from a {view_angle} angle
            - Natural, realistic facial features and expressions
            
            APPAREL DETAILS:
            - The model must be wearing the EXACT apparel from the reference image
            - The apparel should be in {apparel_color_desc}
            - Preserve ALL texture details, patterns, and material properties EXACTLY as shown in the reference image
            - The fit should be natural and appropriate for the model's body type
            
            PHOTOSHOOT SETTINGS:
            - Background: {background}
            - Lighting: {lighting_style}
            - Clean, professional e-commerce/fashion catalog style
            
            CRITICAL: This is a virtual try-on visualization where preserving the EXACT color, texture, and details of
            the uploaded apparel is the highest priority. The apparel must look identical to the reference image.
            """
            
            # Add styling prompt if provided
            if styling_prompt:
                base_prompt += f"\n\nSTYLING INSTRUCTIONS: {styling_prompt}"
            
            # Generate the image with DALL-E 3
            print("Generating try-on image with DALL-E 3...")
            
            # Convert the model image to a base64 string for reference (not used in API call but for debugging)
            if model_image:
                model_image_base64 = self.image_to_base64(model_image)
                print("Model image converted to base64")
            
            # Get quality settings from photoshoot_settings
            dalle_quality = "hd" if photoshoot_settings.get("quality", "Standard") in ["High", "Ultra HD (slower generation)"] else "standard"
            
            # Choose size based on view angle
            dalle_size = "1024x1024"  # Default square
            if view_angle in ["Full body", "Three-quarter front", "Three-quarter back"]:
                dalle_size = "1024x1792"  # Portrait for full body shots
            
            # Generate the final image
            dalle_image = self.generate_image(base_prompt, size=dalle_size, quality=dalle_quality)
            
            if dalle_image:
                # Create a high-quality copy of the fitted image
                high_quality_fitted = dalle_image.copy()
                
                # Create a composite image showing original apparel, model, and final result
                width, height = 800, 1000
                composite = Image.new('RGB', (width, height), color=(245, 245, 250))
                
                # Resize images for the composite
                apparel_resized = apparel_image.copy()
                apparel_resized.thumbnail((width//3 - 20, height//3 - 20))
                
                if model_image:
                    model_resized = model_image.copy()
                    model_resized.thumbnail((width//3 - 20, height//3 - 20))
                    
                fitted_resized = dalle_image.copy()
                fitted_resized.thumbnail((width - 40, height//2))
                
                # Paste images into composite
                composite.paste(apparel_resized, (10, 60))
                
                if model_image:
                    composite.paste(model_resized, (width//3 + 10, 60))
                
                # Center the fitted image in the bottom section
                fit_x = (width - fitted_resized.width) // 2
                composite.paste(fitted_resized, (fit_x, height//3 + 80))
                
                # Add labels using PIL's ImageDraw
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(composite)
                
                # Try to use a nice font, fall back to default if not available
                try:
                    font = ImageFont.truetype("Arial", 16)
                    title_font = ImageFont.truetype("Arial", 20)
                except IOError:
                    font = ImageFont.load_default()
                    title_font = font
                
                # Add captions
                caption = f"Virtual Photoshoot - {pose} Pose - {view_angle}"
                draw.text((width//2 - 200, 20), caption, fill=(0, 0, 0), font=title_font)
                draw.text((10, 40), "Original Apparel", fill=(0, 0, 0), font=font)
                
                if model_image:
                    draw.text((width//3 + 10, 40), "Your Model", fill=(0, 0, 0), font=font)
                    
                draw.text((width//2 - 100, height//3 + 60), "Final Result (DALL-E 3)", fill=(0, 0, 0), font=font)
                
                return composite, "Successfully fitted apparel on model using DALL-E 3", high_quality_fitted
            else:
                return None, "Failed to generate image with DALL-E 3", None
                
        except Exception as e:
            return None, f"Error in DALL-E 3 try-on: {str(e)}", None