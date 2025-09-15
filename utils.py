import google.generativeai as genai
import io
import base64
import requests
import json
import time
import socket
import backoff
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
from typing import Tuple, Dict, Optional, Any, List

# Define a retry decorator for network operations
def retry_with_backoff(retries=3, backoff_factor=1.5):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt == retries:
                        raise e
                    wait_time = backoff_factor * (2 ** (attempt - 1))
                    print(f"Attempt {attempt} failed with error: {e}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
        return wrapper
    return decorator

class ImagenHandler:
    """
    Handler class for Google's Generative AI integration to generate and manipulate images.
    """
    def __init__(self, api_key: str, timeout: int = 60, 
                 imagen_model: str = "models/imagen-4.0-generate-001", 
                 gemini_image_model: str = "gemini-2.5-flash-image-preview"):
        """
        Initialize the ImagenHandler with the Google API key
        
        Args:
            api_key: The Google API key
            timeout: Timeout in seconds for API calls (default: 60)
            imagen_model: Name of the Imagen model to use
            gemini_image_model: Name of the Gemini image model to use
        """
        self.api_key = api_key
        self.timeout = timeout
        self.configure_genai()
        
        # Use Gemini 1.5 Pro model for text generation
        self.text_model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Use the specified image generation models
        self.imagen_model = imagen_model  # For direct image generation
        self.gemini_image_model = genai.GenerativeModel(gemini_image_model)  # For image preview generation
        
        print(f"Initialized with models: Imagen={self.imagen_model}, Gemini={gemini_image_model}")
    
    def configure_genai(self):
        """Configure the genai client with proper timeouts and retries"""
        genai.configure(
            api_key=self.api_key,
            transport="rest",  # Use REST transport which is more reliable than gRPC for some networks
        )
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        """
        buffered = io.BytesIO()
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    @retry_with_backoff(retries=3, backoff_factor=2)
    def generate_image_with_api(self, prompt: str) -> Optional[Image.Image]:
        """
        Generate an image using Google's models with retry logic
        """
        try:
            # Try using the Gemini model for image generation first
            try:
                print(f"Attempting to generate image using Gemini model: {self.gemini_image_model}")
                
                # Use the Gemini model with image generation capability
                try:
                    start_time = time.time()
                    response = self.gemini_image_model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.4,
                        }
                    )
                    
                    # Try to extract image from the response
                    if hasattr(response, 'parts'):
                        for part in response.parts:
                            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                image_bytes = part.inline_data.data
                                image = Image.open(io.BytesIO(image_bytes))
                                print(f"Successfully generated image with Gemini model in {time.time() - start_time:.2f} seconds")
                                return image
                    
                    print("Failed to get image from Gemini image model response")
                    raise Exception("Failed to generate image with Gemini image model")
                    
                except Exception as gemini_error:
                    print(f"Gemini model failed: {gemini_error}")
                    raise gemini_error
                    
            except Exception as all_models_error:
                # If all models fail, fall back to the placeholder
                print(f"All image generation methods failed: {all_models_error}")
                
                # Fall back to the placeholder image generation
                print("Falling back to placeholder image generation")
                width, height = 512, 768
                image = Image.new('RGB', (width, height), color=(245, 245, 250))
                draw = ImageDraw.Draw(image)
                
                # Draw a simple silhouette
                # Head
                draw.ellipse([(width//2-50, 100), (width//2+50, 200)], fill=(200, 200, 210))
                # Body
                draw.polygon([(width//2-70, 200), (width//2+70, 200), 
                          (width//2+100, 500), (width//2-100, 500)], fill=(220, 220, 230))
                # Legs
                draw.rectangle([(width//2-100, 500), (width//2-30, 700)], fill=(210, 210, 220))
                draw.rectangle([(width//2+30, 500), (width//2+100, 700)], fill=(210, 210, 220))
                
                # Add some text indicating this is a generated fashion model
                try:
                    font = ImageFont.truetype("Arial", 18)
                    small_font = ImageFont.truetype("Arial", 12)
                except IOError:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                    
                draw.text((width//2-120, 20), "Fashion Model Visualization", fill=(0, 0, 0), font=font)
                
                # Add error information
                error_msg = f"Error: {str(all_models_error)[:100]}..."
                draw.text((width//2-200, height-180), "Image Generation Failed - Using Placeholder", 
                        fill=(200, 0, 0), font=font)
                draw.text((width//2-200, height-150), error_msg, 
                        fill=(200, 0, 0), font=small_font)
                
                # Add prompt details at the bottom
                wrapped_prompt = textwrap.wrap(prompt, width=60)
                y_text = height - 100
                for i, line in enumerate(wrapped_prompt[:4]):
                    draw.text((10, y_text + i*15), line, fill=(80, 80, 80), font=small_font)
                
                return image
            
        except Exception as e:
            print(f"Image generation completely failed: {e}")
            
            # Create a simple error image
            width, height = 512, 768
            image = Image.new('RGB', (width, height), color=(245, 245, 250))
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("Arial", 18)
                small_font = ImageFont.truetype("Arial", 12)
            except IOError:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
                
            draw.text((width//2-120, height//2), "Image Generation Failed", fill=(200, 0, 0), font=font)
            
            return image
    
    @retry_with_backoff(retries=2, backoff_factor=2)
    def generate_model_image(
        self, 
        gender: str,
        physique: str,
        ethnicity: str,
        height: str,
        hair_color: str,
        hair_style: str,
        skin: str,
        clothing_type: str,
        shoe_style: str,
        facial_expression: str = "Neutral",
        custom_prompt: str = ""
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Generate a base model image with specified attributes using text-to-text generation
        followed by image generation
        
        Args:
            gender: Gender of the model (Male or Female)
            physique: Body type/build of the model
            ethnicity: Ethnicity of the model
            height: Height of the model
            hair_color: Hair color of the model
            hair_style: Hair style of the model
            skin: Skin type of the model
            clothing_type: Type of clothing for the model
            shoe_style: Style of shoes for the model
            facial_expression: Facial expression of the model (Neutral, Smiling, etc.)
            custom_prompt: Additional custom details for the model (accessories, specific features, etc.)
        """
        try:
            # Check if we have a custom prompt with specific overrides
            has_body_type_override = "Body type details:" in custom_prompt if custom_prompt else False
            has_facial_features_override = "Facial features:" in custom_prompt if custom_prompt else False
            has_makeup_override = "Makeup:" in custom_prompt if custom_prompt else False
            has_accessories_override = "Accessories:" in custom_prompt if custom_prompt else False
            has_environment_override = "Environment context:" in custom_prompt if custom_prompt else False
            
            # Start with base attributes that should always be included
            prompt = f"""
            Create a photorealistic fashion model image of a {gender.lower()} {ethnicity.lower()} person 
            who is {height} tall with {hair_color.lower()} {hair_style.lower()} hair and {skin.lower()} skin.
            """
            
            # Only include physique if not overridden by custom prompt
            if not has_body_type_override:
                prompt += f"The model should have a {physique.lower()} build. "
            
            # Add clothing and shoes (these are always kept as they're not typically overridden)
            prompt += f"The model should be in a neutral standing pose, wearing {clothing_type.lower()} and {shoe_style.lower()}, "
            
            # Add facial expression unless it might be overridden in custom prompt
            if not has_facial_features_override and not has_makeup_override:
                prompt += f"with a {facial_expression.lower()} facial expression. "
            else:
                prompt += ". "
                
            # Add standard quality requirements
            prompt += """
            The image should be high-quality, well-lit, suitable for a virtual fitting room application.
            
            Please ensure:
            - The person is centered in the frame with their full body visible
            """
            
            # Add non-overridden attributes
            prompt += f"- The {hair_color.lower()} {hair_style.lower()} hairstyle is clearly visible and accurately represented\n"
            prompt += f"- The {skin.lower()} skin texture is accurately rendered\n"
            prompt += f"- The {clothing_type.lower()} and {shoe_style.lower()} are clearly visible\n"
            prompt += f"- The model MUST be a {gender.lower()} model with appropriate {gender.lower()} physical characteristics\n"
            
            # Add attributes that aren't overridden by custom prompts
            if not has_body_type_override:
                prompt += f"- The model has a {physique.lower()} body type\n"
            
            if not has_facial_features_override and not has_makeup_override:
                prompt += f"- The model has a clear {facial_expression.lower()} expression on their face\n"
            
            # Add background setting if not overridden
            if not has_environment_override:
                prompt += "- Use a simple studio background with soft shadows\n"
            
            # Add custom prompt details as highest priority overrides
            if custom_prompt:
                prompt += f"\n\n### PRIORITY CUSTOM DETAILS (THESE OVERRIDE ANY CONFLICTING INSTRUCTIONS ABOVE) ###\n{custom_prompt}\n"
                prompt += "\nThe above custom details MUST be strictly followed even if they conflict with earlier specifications.\n"
            
            # Get a more detailed description from Gemini
            detail_prompt = f"""
            Refine this image prompt for a text-to-image model to create a photorealistic fashion model:
            '{prompt}'
            
            Add specific details about lighting, pose, clothing specifics, and facial expression.
            Make it highly detailed and optimized for realistic {gender.lower()} fashion model generation.
            
            Emphasize that the model must have:
            1. Exactly {hair_color.lower()} {hair_style.lower()} hair (this is critical)
            2. {skin.lower()} skin as specified
            3. {ethnicity.lower()} ethnicity features
            """
            
            # Add non-overridden elements to the detail prompt
            if not has_body_type_override:
                detail_prompt += f"4. {physique.lower()} body type at {height} tall\n"
            else:
                detail_prompt += f"4. The body type details exactly as specified in the custom prompt, at {height} tall\n"
                
            detail_prompt += f"5. Wearing the exact {clothing_type.lower()} and {shoe_style.lower()}\n"
            
            if not has_facial_features_override and not has_makeup_override:
                detail_prompt += f"6. A clear {facial_expression.lower()} facial expression\n"
            else:
                detail_prompt += "6. The exact facial features and expressions as specified in the custom prompt\n"
            
            # Include custom prompt details as highest priority
            if custom_prompt:
                detail_prompt += f"\n\nTHESE CUSTOM DETAILS ARE HIGHEST PRIORITY AND OVERRIDE ANY CONFLICTING SPECIFICATIONS:\n{custom_prompt}\n"
                detail_prompt += "\nThe model MUST incorporate all these custom details exactly as specified."
            
            detail_prompt += "\n\nThe image should be high-quality fashion photography with perfect lighting."
            
            try:
                response = self.text_model.generate_content(
                    detail_prompt,
                    generation_config={
                        "temperature": 0.4,
                        "max_output_tokens": 1024,
                    }
                )
                
                # Use the enhanced prompt to generate an image
                enhanced_prompt = response.text
                
                # For debugging
                print(f"Final prompt with priority for custom details: {enhanced_prompt[:200]}...")
                
                # Generate the image using our image generation method
                model_image = self.generate_image_with_api(enhanced_prompt)
                
                if model_image:
                    return model_image, enhanced_prompt
                else:
                    return None, "Failed to generate model image"
            except Exception as e:
                print(f"Error in generate_model_image: {e}")
                # If the enhanced prompt generation fails, try with the basic prompt
                model_image = self.generate_image_with_api(prompt)
                if model_image:
                    return model_image, prompt
                else:
                    return None, f"Failed to generate model image: {e}"
            
        except Exception as e:
            return None, str(e)
    
    @retry_with_backoff(retries=2, backoff_factor=2)
    def fit_apparel_on_model(
        self,
        apparel_image: Image.Image,
        model_attributes: Dict[str, str],
        pose: str,
        model_image: Image.Image = None,
        photoshoot_settings: Dict[str, Any] = None,
        styling_prompt: str = "",
        apparel_color: str = None
    ) -> Tuple[Optional[Image.Image], Optional[str], Optional[Image.Image]]:
        """
        Generate an image of apparel fitted on a model with the specified attributes
        
        Args:
            apparel_image: The uploaded apparel image
            model_attributes: Dictionary of model attributes
            pose: The desired pose
            model_image: Optional previously generated model image
            photoshoot_settings: Optional dictionary with photoshoot settings (background, lighting, view angle)
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
                    "lighting_direction": "Front",
                    "quality": "Standard"
                }
            
            # Extract view angle settings
            view_angle = photoshoot_settings.get("view_angle", "Front view")
            
            # Extract background settings
            background = photoshoot_settings.get("background", "Studio white")
            background_desc = self._get_background_description(photoshoot_settings)
            
            # Extract lighting settings
            lighting_style = photoshoot_settings.get("lighting_style", "Standard studio")
            lighting_direction = photoshoot_settings.get("lighting_direction", "Front")
            lighting_desc = self._get_lighting_description(photoshoot_settings)
            
            # Extract quality settings
            quality = photoshoot_settings.get("quality", "Standard")
            quality_factor = 1.0
            if quality == "High":
                quality_factor = 1.5
            elif quality == "Ultra HD (slower generation)":
                quality_factor = 2.0
                
            # Analyze the apparel image to get features
            features = extract_apparel_features(apparel_image)
            
            # Check if we have a facial expression in model attributes
            facial_expression = model_attributes.get("facial_expression", "Neutral")
            
            # Check if the custom prompt contains specific overrides from styling prompt
            has_fit_style_override = "fit" in styling_prompt.lower() if styling_prompt else False
            has_texture_override = "texture" in styling_prompt.lower() if styling_prompt else False
            has_color_override = False  # Will be set to True if apparel_color is provided
            has_drape_override = "drape" in styling_prompt.lower() or "flow" in styling_prompt.lower() if styling_prompt else False
            has_strap_override = "strap" in styling_prompt.lower() or "neckline" in styling_prompt.lower() if styling_prompt else False
            has_sleeve_override = "sleeve" in styling_prompt.lower() if styling_prompt else False
            has_styling_override = styling_prompt != "" if styling_prompt else False
            
            # Update color override flag if apparel_color is provided
            if apparel_color:
                has_color_override = True
            
            # Create a comprehensive description of the model using available model attributes
            model_description_parts = []
            
            # Add basic model attributes
            if 'gender' in model_attributes:
                model_description_parts.append(f"{model_attributes['gender']} model")
            
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
            
            # Get a simplified apparel description
            apparel_description = f"{features['dominant_color']} {features['apparel_type']}"
            if features['pattern_type'] != "solid":
                apparel_description = f"{features['pattern_type']} {apparel_description}"
                
            if apparel_color:
                apparel_description = f"{apparel_color} {features['apparel_type']}"
                
            # Create a prompt that references the specific model already generated
            base_prompt = f"""
            Create a high-resolution photorealistic image of the EXACT SAME {model_attributes.get('gender', '').lower()} model from the reference image, now wearing the uploaded {apparel_description}. 
            
            This is an image editing task: the model is {model_description}, and must now be shown wearing the exact apparel from the uploaded image.
            
            The model should be in a {pose.lower()} pose from a {view_angle.lower()} angle. 
            The model should have exactly the same appearance as the reference image.
            
            IMPORTANT: The model MUST be wearing the apparel shown in the uploaded image. This is CRITICAL.
            
            The model MUST MATCH THE EXACT SAME PERSON shown in the reference image, with:"""
            
            # Add model attributes that are available
            if 'hair_color' in model_attributes and 'hair_style' in model_attributes:
                base_prompt += f"\n- SAME {model_attributes['hair_color'].lower()} {model_attributes['hair_style'].lower()} hairstyle"
            
            if 'ethnicity' in model_attributes and 'skin_tone' in model_attributes:
                base_prompt += f"\n- SAME {model_attributes['ethnicity'].lower()} features and {model_attributes['skin_tone'].lower()}"
            
            if 'build' in model_attributes:
                base_prompt += f"\n- SAME {model_attributes['build'].lower()} body type"
            
            base_prompt += f"""
            
            The apparel from the uploaded image MUST:
            - Be clearly visible on the model
            - Be the primary focus of the image
            - Maintain its original design and shape"""
            
            # Only include color if not overridden by apparel_color
            if not has_color_override:
                base_prompt += f"\n- Keep the original {features['dominant_color']} color exactly as shown"
            else:
                # Add the color change instruction
                base_prompt += f"\n- Be changed to {apparel_color} while preserving all details and patterns"
            
            # Only include pattern if not overridden by custom prompt
            if not has_texture_override:
                base_prompt += f"\n- Preserve the {features['pattern_type']} pattern exactly as shown"
            
            # Only include texture if not overridden by custom prompt
            if not has_texture_override:
                base_prompt += "\n- Maintain the exact fabric texture and material appearance"
            
            # Only include fit if not overridden by custom prompt
            if not has_fit_style_override and not has_drape_override:
                base_prompt += "\n- Fit the model appropriately while maintaining its original proportions"
            
            base_prompt += f"""
            
            Background: {background_desc}
            
            Lighting: {lighting_desc}
            
            The image should be ultra high-quality, well-lit photography with appropriate shadows,
            suitable for an e-commerce or fashion catalog.
            
            CRITICAL: The primary objective is to show the EXACT SAME model from the reference image now wearing
            the EXACT SAME apparel from the uploaded image. This is a virtual try-on visualization.
            """
            
            # Add styling prompt if available, marking it as highest priority
            if styling_prompt:
                base_prompt += f"\n\n### PRIORITY STYLING INSTRUCTIONS (THESE OVERRIDE ANY CONFLICTING INSTRUCTIONS ABOVE) ###\n{styling_prompt}\n"
                base_prompt += "\nThe above styling details MUST be strictly followed even if they conflict with earlier specifications.\n"
            
            # Include both the model image and apparel image in the prompt if available
            multimodal_content = []
            
            # Add the apparel image as reference - make this first for emphasis
            multimodal_content.append({
                "role": "user",
                "parts": [{
                    "text": "This is the EXACT APPAREL that needs to be placed on the model. The model MUST be wearing THIS SPECIFIC ITEM in the final image:"
                }, {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": self.image_to_base64(apparel_image)
                    }
                }]
            })
            
            # Add the model image as reference if available
            if model_image is not None:
                multimodal_content.append({
                    "role": "user",
                    "parts": [{
                        "text": f"This is the exact model who should wear the apparel. Maintain ALL aspects of the model's appearance (hair style, hair color, facial features, body type). This model is: {model_description}"
                    }, {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": self.image_to_base64(model_image)
                        }
                    }]
                })
            
            # If color change is requested, emphasize it again
            color_emphasis = ""
            if apparel_color:  
                color_emphasis = f"\nIMPORTANT: The apparel should be {apparel_color} in color instead of its original color."
            
            # Prepare final prompt combining text and images with stronger emphasis on the task
            prompt_text = f"""
            TASK: Create an image where the model from the reference image is wearing the exact apparel from the uploaded image.
            
            {base_prompt}
            
            Essential requirements:
            1. The model MUST be WEARING the APPAREL from the uploaded image - this is the MOST IMPORTANT requirement
            2. The model's pose should be {pose.lower()}
            3. The model should have a {facial_expression.lower()} facial expression
            4. The view angle should be {view_angle.lower()}
            {color_emphasis}
            
            Make absolutely certain that:
            - The apparel from the uploaded image is CLEARLY VISIBLE on the model
            - The apparel fits the model appropriately
            - The model maintains their exact appearance from the reference image
            
            This is a virtual try-on visualization for e-commerce where customers want to see how the specific apparel item will look when worn.
            """
            
            # Only add generic fit instructions if not overridden
            if not has_fit_style_override and not has_drape_override:
                prompt_text += "\n- Ensure the apparel is clearly visible and fitted naturally\n"
                
            # Add styling prompt to final prompt text if available
            if styling_prompt:
                prompt_text += f"\n\nSTRICTLY APPLY THESE CUSTOM STYLING AND FIT DETAILS (HIGHEST PRIORITY):\n{styling_prompt}\n"
                prompt_text += "\nThese styling details OVERRIDE any conflicting instructions above."
            
            # Add a final instruction to emphasize the critical requirement
            prompt_text += "\n\nCRITICAL REMINDER: The final image MUST show the model WEARING the apparel from the uploaded image."
            
            multimodal_content.append({
                "role": "user",
                "parts": [{
                    "text": prompt_text
                }]
            })
            
            # For debugging
            print(f"Using styling prompt: {styling_prompt[:100]}..." if styling_prompt else "No styling prompt")
            
            # Try the gemini model for apparel fitting using multimodal input
            try:
                print("Using Gemini model for apparel fitting with multimodal input")
                
                # First approach: Use multimodal input if available
                if model_image is not None:
                    # Create a conversation that includes both images with higher temperature for creativity
                    gemini_response = self.gemini_image_model.generate_content(
                        multimodal_content,
                        generation_config={
                            "temperature": 0.7,  # Increased from 0.2 to allow more creative freedom
                        }
                    )
                    
                    # Extract image from response
                    fitted_image = None
                    if hasattr(gemini_response, 'parts'):
                        for part in gemini_response.parts:
                            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                try:
                                    # First try with direct data access
                                    image_bytes = part.inline_data.data
                                    fitted_image = Image.open(io.BytesIO(image_bytes))
                                except Exception as img_error:
                                    try:
                                        # If that fails, try with base64 decoding
                                        image_bytes = base64.b64decode(part.inline_data.data)
                                        fitted_image = Image.open(io.BytesIO(image_bytes))
                                    except Exception as decode_error:
                                        print(f"Failed to decode image: {decode_error}")
                                        raise decode_error
                                
                                print("Successfully generated fitted image with multimodal input")
                                
                                # Save the original high-quality fitted image
                                high_quality_fitted = fitted_image.copy()
                                
                                # Create composite image showing original apparel and fitted result
                                width, height = 800, 1000
                                composite = Image.new('RGB', (width, height), color=(245, 245, 250))
                                
                                # Add all three images: apparel, model, and final result
                                # Resize images
                                apparel_resized = apparel_image.copy()
                                apparel_resized.thumbnail((width//3 - 20, height//3 - 20))
                                
                                model_resized = model_image.copy()
                                model_resized.thumbnail((width//3 - 20, height//3 - 20))
                                
                                fitted_resized = fitted_image.copy()
                                fitted_resized.thumbnail((width - 40, height//2))
                                
                                # Calculate positions
                                composite.paste(apparel_resized, (10, 60))
                                composite.paste(model_resized, (width//3 + 10, 60))
                                
                                # Center the fitted image in the bottom section
                                fit_x = (width - fitted_resized.width) // 2
                                composite.paste(fitted_resized, (fit_x, height//3 + 80))
                                
                                # Add labels
                                draw = ImageDraw.Draw(composite)
                                try:
                                    font = ImageFont.truetype("Arial", 16)
                                    title_font = ImageFont.truetype("Arial", 20)
                                except IOError:
                                    font = ImageFont.load_default()
                                    title_font = font
                                
                                caption = f"Virtual Photoshoot - {pose} Pose - {view_angle}"
                                draw.text((width//2 - 200, 20), caption, fill=(0, 0, 0), font=title_font)
                                draw.text((10, 40), "Original Apparel", fill=(0, 0, 0), font=font)
                                draw.text((width//3 + 10, 40), "Your Model", fill=(0, 0, 0), font=font)
                                draw.text((width//2 - 100, height//3 + 60), "Final Result", fill=(0, 0, 0), font=font)
                                
                                return composite, "Successfully fitted apparel on model", high_quality_fitted

                    # If we get here, no image was found in the response
                    print("No image found in Gemini response")
                    return None, "Failed to generate image with apparel fitted on model", None

                else:
                    return None, "Model image is required for fitting apparel", None

            except Exception as multimodal_error:
                print(f"Multimodal approach failed: {multimodal_error}")
                return None, str(multimodal_error), None
            
        except Exception as e:
            return None, str(e), None

    def _get_background_description(self, photoshoot_settings: Dict[str, Any]) -> str:
        """Get a detailed description of the background based on photoshoot settings"""
        background = photoshoot_settings.get("background", "Studio white")
        
        if background == "Studio white":
            return "Clean white studio background with subtle shadows"
        elif background == "Studio light gray":
            return "Light gray studio background with soft shadows"
        elif background == "Studio dark gray":
            return "Dark gray studio background with defined shadows"
        elif background == "Studio black":
            return "Deep black studio background with dramatic lighting"
        elif background == "Minimalist interior":
            return "Minimalist interior with neutral tones and clean lines"
        elif background == "Urban street":
            return "Stylish urban street setting with soft bokeh effect"
        elif background == "Nature outdoor":
            return "Soft focus natural outdoor setting with blurred trees and foliage"
        elif background == "Beach":
            return "Beach setting with soft sand and out of focus ocean"
        elif background == "Gradient":
            # Handle gradient settings
            direction = photoshoot_settings.get("gradient_direction", "Top to Bottom")
            colors = photoshoot_settings.get("gradient_colors", "Blue to Purple")
            
            if colors == "Custom" and "gradient_color1" in photoshoot_settings:
                color1 = photoshoot_settings.get("gradient_color1", "#ffffff")
                color2 = photoshoot_settings.get("gradient_color2", "#e0e0e0")
                return f"Custom gradient background from {color1} to {color2}, direction: {direction}"
            else:
                return f"Smooth gradient background with {colors} colors, direction: {direction}"
        elif background == "Solid color":
            color = photoshoot_settings.get("background_color", "#f0f0f0")
            return f"Solid color background ({color})"
        elif background == "Transparent (for e-commerce)":
            return "Perfectly transparent background with no shadows, suitable for e-commerce product listings"
        else:
            return "Clean studio background"
    
    def _get_lighting_description(self, photoshoot_settings: Dict[str, Any]) -> str:
        """Get a detailed description of the lighting based on photoshoot settings"""
        style = photoshoot_settings.get("lighting_style", "Standard studio")
        direction = photoshoot_settings.get("lighting_direction", "Front")
        
        # Get advanced lighting controls if available
        intensity = photoshoot_settings.get("light_intensity", 0.7)
        contrast = photoshoot_settings.get("contrast", 0.5)
        shadows = photoshoot_settings.get("shadows", 0.3)
        
        # Convert numeric values to descriptive text
        intensity_desc = "medium"
        if intensity < 0.4: intensity_desc = "low"
        elif intensity > 0.7: intensity_desc = "high"
        
        contrast_desc = "medium"
        if contrast < 0.4: contrast_desc = "low"
        elif contrast > 0.7: contrast_desc = "high"
        
        shadow_desc = "medium"
        if shadows < 0.4: shadow_desc = "soft"
        elif shadows > 0.7: shadow_desc = "dark"
        
        # Build lighting description
        if style == "Standard studio":
            return f"{direction} lighting with {intensity_desc} intensity, {contrast_desc} contrast, and {shadow_desc} shadows"
        elif style == "Soft diffused":
            return f"Soft diffused {direction} lighting with gentle shadows and smooth transitions"
        elif style == "Dramatic":
            return f"Dramatic {direction} lighting with strong contrast and deep shadows"
        elif style == "Bright high-key":
            return f"Bright high-key lighting from {direction} with minimal shadows and high brightness"
        elif style == "Dark low-key":
            return f"Dark low-key {direction} lighting with deep shadows and dramatic contrast"
        elif style == "Natural sunlight":
            return f"Natural sunlight from {direction} with soft shadows and warm tone"
        elif style == "Golden hour":
            return f"Warm golden hour lighting from {direction} with long shadows and orange-gold tint"
        elif style == "Blue hour":
            return f"Cool blue hour lighting from {direction} with soft diffusion and blue-purple tint"
        else:
            return f"{direction} lighting with {intensity_desc} intensity"

    def generate_photoshoot_variations(
        self,
        apparel_image: Image.Image,
        model_attributes: Dict[str, str],
        variations: List[str],
        count: int = 3,
        model_image: Image.Image = None,
        variation_type: str = "pose",
        photoshoot_settings: Dict[str, Any] = None
    ) -> Tuple[List[Image.Image], Optional[str]]:
        """
        Generate multiple photoshoot variations with different settings
        
        Args:
            apparel_image: The uploaded apparel image
            model_attributes: Dictionary of model attributes
            variations: List of variations (poses, backgrounds, or view angles)
            count: Number of variations to generate
            model_image: Optional previously generated model image
            variation_type: Type of variation ("pose", "background", or "view")
            photoshoot_settings: Base photoshoot settings
        """
        try:
            results = []
            
            # Set default photoshoot settings if not provided
            if photoshoot_settings is None:
                photoshoot_settings = {
                    "pose": "Natural standing",
                    "view_angle": "Front view",
                    "background": "Studio white",
                    "lighting_style": "Standard studio",
                    "lighting_direction": "Front",
                    "quality": "Standard"
                }
                
            # Make a copy of the base settings that we can modify for each variation
            base_settings = dict(photoshoot_settings)
            
            for i in range(min(count, len(variations))):
                # Create variation-specific settings
                variation_settings = dict(base_settings)
                
                if variation_type == "pose":
                    # For pose variations, we update the pose in settings
                    variation_pose = variations[i]
                    variation_view = base_settings.get("view_angle", "Front view")
                    variation_background = base_settings.get("background", "Studio white")
                elif variation_type == "background":
                    # For background variations, we update the background in settings
                    variation_settings["background"] = variations[i]
                    variation_pose = base_settings.get("pose", "Natural standing")
                    variation_view = base_settings.get("view_angle", "Front view")
                    variation_background = variations[i]
                elif variation_type == "view":
                    # For view angle variations, we update the view angle in settings
                    variation_settings["view_angle"] = variations[i]
                    variation_pose = base_settings.get("pose", "Natural standing")
                    variation_view = variations[i]
                    variation_background = base_settings.get("background", "Studio white")
                else:
                    # Default to pose variations
                    variation_pose = variations[i]
                    variation_view = base_settings.get("view_angle", "Front view")
                    variation_background = base_settings.get("background", "Studio white")
                
                # Generate this variation
                image, error, _ = self.fit_apparel_on_model(
                    apparel_image,
                    model_attributes,
                    variation_pose,
                    model_image=model_image,
                    photoshoot_settings=variation_settings
                )
                
                if image:
                    results.append(image)
                
            if not results:
                return [], "Failed to generate any photoshoot variations"
            
            return results, None
            
        except Exception as e:
            return [], str(e)

# Additional utility functions for image processing

def extract_apparel_features(image: Image.Image) -> Dict[str, Any]:
    """
    Extract key features from apparel image to better describe it in prompts
    """
    # In a real implementation, we would use computer vision to analyze the image
    # For this prototype, we'll return placeholder values
    
    # Get average color of the image as a simple analysis
    avg_color = get_average_color(image)
    
    # Convert RGB to color name (simplified)
    color_name = get_color_name(avg_color)
    
    # Simplified pattern detection
    pattern = "solid"  # Default
    
    # Simple apparel type detection based on image dimensions
    width, height = image.size
    aspect_ratio = height / width if width > 0 else 0
    
    if aspect_ratio > 1.5:
        apparel_type = "dress or long top"
    elif aspect_ratio > 1.0:
        apparel_type = "shirt or top"
    else:
        apparel_type = "clothing item"
    
    return {
        "dominant_color": color_name,
        "pattern_type": pattern,
        "apparel_type": apparel_type
    }

def get_average_color(image: Image.Image) -> tuple:
    """Get the average color of an image"""
    # Resize image to speed up processing
    img_resized = image.resize((50, 50))
    # Convert to RGB if not already
    if img_resized.mode != "RGB":
        img_resized = img_resized.convert("RGB")
    
    # Get average color
    pixels = list(img_resized.getdata())
    r_total = sum(pixel[0] for pixel in pixels)
    g_total = sum(pixel[1] for pixel in pixels)
    b_total = sum(pixel[2] for pixel in pixels)
    pixel_count = len(pixels)
    
    return (r_total // pixel_count, g_total // pixel_count, b_total // pixel_count)

def get_color_name(rgb: tuple) -> str:
    """Simple function to convert RGB to a color name"""
    r, g, b = rgb
    
    # Very simplified color mapping
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > g and r > b:
        if g > 150 and b < 100:
            return "yellow"
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    elif abs(r - g) < 30 and abs(r - b) < 30 and abs(g - b) < 30:
        if r > 150:
            return "light gray"
        else:
            return "dark gray"
    else:
        return "multicolor"