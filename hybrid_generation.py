import io
import base64
import requests
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat
from typing import Tuple, Dict, Optional, Any, List
import backoff
import colorsys
from collections import Counter

class HybridApparelFittingHandler:
    """
    A hybrid approach that uses OpenAI to extract apparel features and Google's Gemini for the actual try-on.
    This helps preserve the exact color and texture of apparel while maintaining model consistency.
    """
    def __init__(self, openai_api_key: str, google_api_key: str, timeout: int = 60):
        """
        Initialize the HybridApparelFittingHandler with both API keys
        
        Args:
            openai_api_key: The OpenAI API key for feature extraction
            google_api_key: The Google API key for image generation
            timeout: Timeout in seconds for API calls (default: 60)
        """
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.timeout = timeout
        
        # OpenAI API settings
        self.openai_base_url = "https://api.openai.com/v1"
        self.openai_headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize Google's Gemini handler for image generation
        # We'll import this here to avoid circular imports
        from utils import ImagenHandler
        self.imagen_handler = ImagenHandler(
            google_api_key, 
            timeout=timeout,
            imagen_model="models/imagen-4.0-generate-001",
            gemini_image_model="gemini-2.5-flash-image-preview"
        )
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def extract_dominant_colors(self, image: Image.Image, num_colors: int = 3) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Extract the dominant colors from an image with their percentages
        This improved version focuses on central regions of the image to avoid background influence
        
        Args:
            image: PIL Image object
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of tuples ((r,g,b), percentage) of dominant colors
        """
        # Resize image to speed up processing
        img_resized = image.copy()
        img_resized.thumbnail((200, 200))
        
        # Convert to RGB if not already
        if img_resized.mode != "RGB":
            img_resized = img_resized.convert("RGB")
            
        # Get dimensions
        width, height = img_resized.size
        
        # Create a mask to prioritize the central region (where the main apparel usually is)
        # and de-emphasize the edges (which might contain background)
        center_x, center_y = width // 2, height // 2
        center_region_size = min(width, height) // 2
        
        # Extract pixels from the center region which is more likely to contain the actual apparel
        pixels = []
        for y in range(height):
            for x in range(width):
                # Calculate distance from center
                dist_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                # Include pixel if it's in the central region (with higher weight)
                if dist_from_center < center_region_size:
                    # Get pixel color
                    pixel = img_resized.getpixel((x, y))
                    
                    # Add pixel multiple times based on proximity to center (weighted sampling)
                    weight = int(10 * (1 - dist_from_center / center_region_size))
                    pixels.extend([pixel] * max(1, weight))
        
        # Count occurrences of each color
        color_count = Counter(pixels)
        total_pixels = len(pixels)
        
        # Get top n colors
        dominant_colors = color_count.most_common(num_colors)
        
        # Filter out white, near-white, black and near-black colors
        filtered_colors = []
        for color, count in dominant_colors:
            r, g, b = color
            # Check if the color is not white/near-white or black/near-black
            if not ((r > 230 and g > 230 and b > 230) or (r < 25 and g < 25 and b < 25)):
                filtered_colors.append((color, count))
        
        # If we filtered out all colors, just return the original dominant colors
        if not filtered_colors:
            filtered_colors = dominant_colors
        
        # Convert to percentage
        dominant_colors_with_percent = [
            (color, count / total_pixels * 100) 
            for color, count in filtered_colors[:num_colors]
        ]
        
        return dominant_colors_with_percent

    def get_mean_color(self, image: Image.Image) -> Tuple[int, int, int]:
        """
        Get the mean color of an image, focusing on the central region
        
        Args:
            image: PIL Image object
            
        Returns:
            (r,g,b) tuple of the mean color
        """
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Crop to central region to avoid background influence
        width, height = image.size
        crop_size = min(width, height) // 2
        center_x, center_y = width // 2, height // 2
        
        # Crop around center
        cropped = image.crop((
            center_x - crop_size//2,
            center_y - crop_size//2,
            center_x + crop_size//2,
            center_y + crop_size//2
        ))
        
        # Use ImageStat to calculate mean color
        stat = ImageStat.Stat(cropped)
        mean_color = tuple(int(x) for x in stat.mean)
        
        return mean_color
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Get a descriptive name for a color based on its RGB values
        Uses HSV transformation for better color description
        """
        r, g, b = rgb
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h = h * 360  # Convert hue to degrees
        
        # Determine if the color is grayscale
        if s < 0.1:
            if v < 0.2:
                return "Black"
            elif v < 0.5:
                return "Dark Gray"
            elif v < 0.8:
                return "Gray"
            else:
                return "White"
                
        # Determine the hue name
        hue_name = ""
        if h < 15 or h >= 345:
            hue_name = "Red"
        elif h < 45:
            hue_name = "Orange"
        elif h < 75:
            hue_name = "Yellow"
        elif h < 105:
            hue_name = "Yellow-Green"
        elif h < 135:
            hue_name = "Green"
        elif h < 165:
            hue_name = "Blue-Green"
        elif h < 195:
            hue_name = "Cyan"
        elif h < 225:
            hue_name = "Light Blue"
        elif h < 255:
            hue_name = "Blue"
        elif h < 285:
            hue_name = "Purple"
        elif h < 315:
            hue_name = "Magenta"
        else:
            hue_name = "Pink"
            
        # Add lightness/darkness
        if v < 0.3:
            lightness = "Dark "
        elif v > 0.8:
            lightness = "Bright "
        else:
            lightness = ""
            
        # Add saturation
        if 0.1 <= s < 0.3:
            saturation = "Muted "
        elif s > 0.8:
            saturation = "Vibrant "
        else:
            saturation = ""
            
        return f"{lightness}{saturation}{hue_name}"
    
    def extract_texture_pattern(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract texture and pattern information from an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with texture and pattern information
        """
        # Resize image to speed up processing
        img_resized = image.copy()
        img_resized.thumbnail((200, 200))
        
        # Convert to grayscale
        if img_resized.mode != "L":
            img_gray = img_resized.convert("L")
        else:
            img_gray = img_resized
        
        # Convert to numpy array
        img_array = np.array(img_gray)
        
        # Calculate standard deviation as a measure of texture complexity
        texture_complexity = np.std(img_array)
        
        # Calculate gradient as a measure of edge/pattern density
        gradient_x = np.gradient(img_array, axis=1)
        gradient_y = np.gradient(img_array, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        pattern_density = np.mean(gradient_magnitude)
        
        # Determine texture type based on complexity
        if texture_complexity < 10:
            texture_type = "smooth"
        elif texture_complexity < 30:
            texture_type = "slightly textured"
        elif texture_complexity < 50:
            texture_type = "moderately textured"
        else:
            texture_type = "highly textured"
            
        # Determine pattern type based on density
        if pattern_density < 5:
            pattern_type = "solid"
        elif pattern_density < 15:
            pattern_type = "subtle pattern"
        elif pattern_density < 30:
            pattern_type = "distinct pattern"
        else:
            pattern_type = "complex pattern"
            
        return {
            "texture_type": texture_type,
            "texture_complexity": float(texture_complexity),
            "pattern_type": pattern_type,
            "pattern_density": float(pattern_density)
        }
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, 
                          requests.exceptions.Timeout, 
                          requests.exceptions.ConnectionError),
                         max_tries=3)
    def extract_apparel_features_with_openai(self, apparel_image: Image.Image) -> Dict[str, Any]:
        """
        Use OpenAI's vision capabilities to extract detailed features from the apparel image
        
        Args:
            apparel_image: The uploaded apparel image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            print("Extracting apparel features with OpenAI...")
            
            # First, get the mean color of the central region
            mean_color_rgb = self.get_mean_color(apparel_image)
            mean_color_hex = self.rgb_to_hex(mean_color_rgb)
            
            # Also get dominant colors with weighted central region sampling
            dominant_colors = self.extract_dominant_colors(apparel_image, num_colors=5)
            
            # Use the most dominant non-white color as primary
            primary_color_rgb = dominant_colors[0][0]
            primary_color_hex = self.rgb_to_hex(primary_color_rgb)
            primary_color_name = self.get_color_name(primary_color_rgb)
            primary_color_percent = dominant_colors[0][1]
            
            # Compare mean and dominant colors - if they're significantly different, 
            # prefer the mean color as it's less likely to be influenced by outliers
            r1, g1, b1 = mean_color_rgb
            r2, g2, b2 = primary_color_rgb
            color_distance = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)**0.5
            
            # If mean and dominant are close, use dominant; otherwise use mean
            final_color_rgb = primary_color_rgb if color_distance < 50 else mean_color_rgb
            final_color_hex = self.rgb_to_hex(final_color_rgb)
            final_color_name = self.get_color_name(final_color_rgb)
            
            print(f"Extracted color: {final_color_name} ({final_color_hex})")
            
            # Get secondary colors (excluding the primary color)
            secondary_colors = []
            for color_rgb, percent in dominant_colors[1:]:
                if percent > 5.0:  # Only include colors that make up more than 5% of the image
                    # Calculate distance from primary color to ensure diversity
                    r1, g1, b1 = final_color_rgb
                    r2, g2, b2 = color_rgb
                    color_dist = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)**0.5
                    
                    # Only include if sufficiently different from primary
                    if color_dist > 30:
                        secondary_colors.append({
                            "rgb": color_rgb,
                            "hex": self.rgb_to_hex(color_rgb),
                            "name": self.get_color_name(color_rgb),
                            "percentage": percent
                        })
            
            # Extract texture and pattern information
            texture_info = self.extract_texture_pattern(apparel_image)
            
            # Save a debug image with the extracted color
            debug_img = Image.new('RGB', (100, 100), final_color_rgb)
            debug_img.save("/tmp/extracted_color.png")
            print(f"Saved debug image with extracted color to /tmp/extracted_color.png")
            
            # Convert image to base64
            base64_image = self.image_to_base64(apparel_image)
            
            # Call OpenAI API to extract features using GPT-4 Vision
            url = f"{self.openai_base_url}/chat/completions"
            
            # Include computational analysis results in the prompt
            analysis_prompt = f"""
            I've computationally analyzed this apparel image and extracted its EXACT color:
            - Primary color: {final_color_name} ({final_color_hex})
            - Texture type: {texture_info['texture_type']} (complexity: {texture_info['texture_complexity']:.1f})
            - Pattern type: {texture_info['pattern_type']} (density: {texture_info['pattern_density']:.1f})
            
            I need to ensure the EXACT original color is preserved in the virtual try-on.
            
            Analyze this apparel image in detail and provide a JSON object with the following properties:
            1. apparel_type (e.g., hoodie, t-shirt, sweater, etc.)
            2. dominant_color - Use EXACTLY "{final_color_name} ({final_color_hex})"
            3. secondary_colors (array of any other notable colors)
            4. pattern_description (detailed pattern description)
            5. texture_description (detailed description of fabric texture)
            6. special_features (array of unique features like pockets, zippers, etc.)
            7. fit_type (e.g., slim, regular, oversized)
            8. sleeve_type (e.g., short sleeve, long sleeve, sleeveless)
            9. neckline_type (e.g., crew neck, v-neck, hood, etc.)
            10. fabric_type (best guess at material)
            11. color_hex_exact (MUST be exactly "{final_color_hex}")
            
            CRITICAL: The color_hex_exact MUST be exactly {final_color_hex} for accurate color reproduction.
            This is absolutely essential for our virtual try-on application.
            
            Provide ONLY the JSON object without any additional text.
            """
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert fashion analyst with deep knowledge of apparel, fabrics, patterns, and textures. Your task is to analyze an image of apparel and extract key features, ensuring the EXACT original color is preserved. Color accuracy is absolutely critical."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(
                url,
                headers=self.openai_headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the JSON from the text response
                content = response_data["choices"][0]["message"]["content"]
                
                # Find the start and end of the JSON object
                try:
                    # First try to parse as is, in case it's clean JSON
                    features = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from the text
                    try:
                        # Look for JSON between code blocks
                        json_start = content.find("```json")
                        if json_start != -1:
                            json_start += 7  # Length of "```json"
                            json_end = content.find("```", json_start)
                            if json_end != -1:
                                features = json.loads(content[json_start:json_end].strip())
                            else:
                                raise ValueError("Could not find end of JSON block")
                        else:
                            # Try to find JSON without code block markers
                            json_start = content.find("{")
                            json_end = content.rfind("}") + 1
                            if json_start != -1 and json_end > json_start:
                                features = json.loads(content[json_start:json_end])
                            else:
                                raise ValueError("Could not find JSON object in response")
                    except Exception as json_error:
                        print(f"Error parsing JSON: {json_error}")
                        # Fallback to basic features with computational analysis
                        features = {
                            "apparel_type": "garment",
                            "dominant_color": final_color_name,
                            "secondary_colors": [color["name"] for color in secondary_colors],
                            "pattern_description": texture_info["pattern_type"],
                            "texture_description": f"{texture_info['texture_type']} fabric texture",
                            "special_features": [],
                            "fit_type": "regular",
                            "sleeve_type": "unknown",
                            "neckline_type": "unknown",
                            "fabric_type": "unknown",
                            "color_temperature": "neutral",
                            "color_hex_exact": final_color_hex
                        }
                
                # CRITICAL: Ensure the computational color extraction ALWAYS overrides any AI suggestion
                # This ensures the exact color from the image is preserved
                features["color_hex_exact"] = final_color_hex
                features["dominant_color"] = f"{final_color_name} ({final_color_hex})"
                
                # Store the original RGB values for reference
                features["color_rgb_exact"] = final_color_rgb
                
                # Ensure we have at least the minimum required fields
                required_fields = ["apparel_type", "dominant_color", "pattern_description", "texture_description"]
                for field in required_fields:
                    if field not in features:
                        features[field] = "unknown"
                
                # Add computational analysis results for reference
                features["computational_analysis"] = {
                    "primary_color": {
                        "rgb": final_color_rgb,
                        "hex": final_color_hex,
                        "name": final_color_name
                    },
                    "secondary_colors": secondary_colors,
                    "texture_info": texture_info
                }
                
                print(f"Extracted features: {features}")
                return features
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                print(error_msg)
                
                # Return basic features based on computational analysis
                return {
                    "apparel_type": "garment",
                    "dominant_color": f"{final_color_name} ({final_color_hex})",
                    "secondary_colors": [color["name"] for color in secondary_colors],
                    "pattern_description": texture_info["pattern_type"],
                    "texture_description": f"{texture_info['texture_type']} fabric texture",
                    "special_features": [],
                    "fit_type": "regular",
                    "sleeve_type": "unknown",
                    "neckline_type": "unknown",
                    "fabric_type": "unknown",
                    "color_temperature": "neutral",
                    "color_hex_exact": final_color_hex,
                    "color_rgb_exact": final_color_rgb,
                    "computational_analysis": {
                        "primary_color": {
                            "rgb": final_color_rgb,
                            "hex": final_color_hex,
                            "name": final_color_name
                        },
                        "secondary_colors": secondary_colors,
                        "texture_info": texture_info
                    }
                }
                
        except Exception as e:
            print(f"Error extracting apparel features with OpenAI: {str(e)}")
            # Try to still return color information if possible
            try:
                # Extract colors computationally as fallback
                mean_color_rgb = self.get_mean_color(apparel_image)
                mean_color_hex = self.rgb_to_hex(mean_color_rgb)
                mean_color_name = self.get_color_name(mean_color_rgb)
                
                texture_info = self.extract_texture_pattern(apparel_image)
                
                return {
                    "apparel_type": "garment",
                    "dominant_color": f"{mean_color_name} ({mean_color_hex})",
                    "secondary_colors": [],
                    "pattern_description": texture_info["pattern_type"],
                    "texture_description": f"{texture_info['texture_type']} fabric texture",
                    "special_features": [],
                    "fit_type": "regular",
                    "sleeve_type": "unknown",
                    "neckline_type": "unknown",
                    "fabric_type": "unknown",
                    "color_temperature": "neutral",
                    "color_hex_exact": mean_color_hex,
                    "color_rgb_exact": mean_color_rgb,
                    "computational_analysis": {
                        "primary_color": {
                            "rgb": mean_color_rgb,
                            "hex": mean_color_hex,
                            "name": mean_color_name
                        },
                        "texture_info": texture_info
                    }
                }
            except:
                # Ultimate fallback
                return {
                    "apparel_type": "garment",
                    "dominant_color": "unknown",
                    "secondary_colors": [],
                    "pattern_description": "solid",
                    "texture_description": "standard fabric texture",
                    "special_features": [],
                    "fit_type": "regular",
                    "sleeve_type": "unknown",
                    "neckline_type": "unknown",
                    "fabric_type": "unknown",
                    "color_hex_exact": "#0000FF", # Use blue as emergency fallback for visibility
                    "color_rgb_exact": (0, 0, 255)
                }
    
    def create_enhanced_styling_prompt(self, 
                                      apparel_features: Dict[str, Any],
                                      base_styling: str = "",
                                      apparel_color: str = None) -> str:
        """
        Create an enhanced styling prompt based on extracted apparel features
        
        Args:
            apparel_features: Dictionary of extracted features
            base_styling: Optional base styling prompt from user
            apparel_color: Optional color to change the apparel to (hex code)
            
        Returns:
            Enhanced styling prompt for Gemini
        """
        # Start with basic information about the apparel
        styling_parts = []
        
        # Get the precise color hex code from features or override
        color_hex = apparel_color if apparel_color else apparel_features.get('color_hex_exact', "#0000FF")
        color_description = apparel_features.get('dominant_color', 'unknown color')
        
        # Add user's styling instructions with high emphasis if provided
        if base_styling:
            styling_parts.append(f"USER'S EXACT STYLING INSTRUCTIONS - FOLLOW PRECISELY:\n{base_styling}")
            styling_parts.append("==== THE ABOVE STYLING INSTRUCTIONS MUST BE FOLLOWED WITH HIGHEST PRIORITY ====")
        
        # Add EXACT COLOR REPRODUCTION as high priority instruction
        # Use strong emphasis and repetition for critical instructions
        styling_parts.append(f"HIGHEST PRIORITY: COLOR MATCH. The {apparel_features['apparel_type']} MUST be EXACTLY {color_description}. The exact hex value is {color_hex}. Do NOT change this color. Do NOT convert to white or any other color.")
        styling_parts.append(f"MANDATORY COLOR INSTRUCTION: The apparel must match the exact RGB values: {apparel_features.get('color_rgb_exact', (0,0,255))}. This is NON-NEGOTIABLE.")
        
        # Add pattern information with enhanced detail
        pattern_desc = apparel_features.get('pattern_description', 'solid')
        if pattern_desc != "solid":
            styling_parts.append(f"The {apparel_features['apparel_type']} has a {pattern_desc} pattern that must be precisely preserved.")
        
        # Add detailed texture information
        texture_desc = apparel_features.get('texture_description', 'standard texture')
        styling_parts.append(f"The fabric texture is {texture_desc} and should be accurately reproduced.")
        
        # Add fabric type if available
        if 'fabric_type' in apparel_features and apparel_features['fabric_type'] != "unknown":
            styling_parts.append(f"The fabric appears to be {apparel_features['fabric_type']}, which affects how light interacts with the surface.")
        
        # Add information about special features
        if 'special_features' in apparel_features and apparel_features['special_features']:
            if isinstance(apparel_features['special_features'], list):
                features_str = ", ".join(apparel_features['special_features'])
                styling_parts.append(f"The garment includes these special features: {features_str}.")
            elif isinstance(apparel_features['special_features'], str):
                styling_parts.append(f"The garment includes these special features: {apparel_features['special_features']}.")
        
        # Add information about fit
        if 'fit_type' in apparel_features and apparel_features['fit_type'] != "unknown":
            styling_parts.append(f"The {apparel_features['apparel_type']} has a {apparel_features['fit_type']} fit style.")
        
        # Add sleeve information
        if 'sleeve_type' in apparel_features and apparel_features['sleeve_type'] != "unknown":
            styling_parts.append(f"It has {apparel_features['sleeve_type']} sleeves.")
        
        # Add neckline information
        if 'neckline_type' in apparel_features and apparel_features['neckline_type'] != "unknown":
            styling_parts.append(f"The neckline is a {apparel_features['neckline_type']}.")
        
        # Add secondary colors if available
        if 'secondary_colors' in apparel_features and apparel_features['secondary_colors']:
            if isinstance(apparel_features['secondary_colors'], list) and len(apparel_features['secondary_colors']) > 0:
                if isinstance(apparel_features['secondary_colors'][0], str):
                    colors_str = ", ".join(apparel_features['secondary_colors'])
                    styling_parts.append(f"The apparel also includes these secondary colors: {colors_str}.")
                elif isinstance(apparel_features['secondary_colors'][0], dict):
                    colors = [f"{c.get('name', 'unknown')} ({c.get('hex', '#000000')})" for c in apparel_features['secondary_colors']]
                    colors_str = ", ".join(colors)
                    styling_parts.append(f"The apparel also includes these secondary colors: {colors_str}.")
            elif isinstance(apparel_features['secondary_colors'], str):
                styling_parts.append(f"The apparel also includes these secondary colors: {apparel_features['secondary_colors']}.")
        
        # Add computational analysis information for extra precision
        if 'computational_analysis' in apparel_features:
            comp_analysis = apparel_features['computational_analysis']
            if 'primary_color' in comp_analysis:
                primary = comp_analysis['primary_color']
                rgb_str = f"RGB({primary.get('rgb', (0,0,0))[0]}, {primary.get('rgb', (0,0,0))[1]}, {primary.get('rgb', (0,0,0))[2]})"
                styling_parts.append(f"TECHNICAL COLOR INFO: Primary color is {primary.get('name', 'unknown')} with hex code {primary.get('hex', '#000000')} and {rgb_str}.")
            
            if 'texture_info' in comp_analysis:
                texture = comp_analysis['texture_info']
                styling_parts.append(f"TECHNICAL TEXTURE INFO: The fabric is {texture.get('texture_type', 'unknown')} with a complexity score of {texture.get('texture_complexity', 0):.1f} and pattern density of {texture.get('pattern_density', 0):.1f}.")
        
        # Add multiple reminders about color accuracy at different points in the prompt
        styling_parts.append(f"COLOR CHECK 1: The {apparel_features['apparel_type']} must be {color_description} with the exact hex code {color_hex}. Do NOT lighten, whiten, or change this color.")
        
        # If user provided styling instructions, remind again after the technical details
        if base_styling:
            styling_parts.append(f"REMINDER OF USER'S STYLING INSTRUCTIONS:\n{base_styling}")
            
        # Final color reminder at the end of the prompt
        styling_parts.append(f"FINAL COLOR CHECK: The apparel color must be EXACTLY {color_hex}. This is absolutely critical for accurate virtual try-on. The model must be wearing an apparel with this exact color - NOT white, NOT lightened, EXACTLY this color: {color_hex}.")
        
        # Combine all styling parts into one enhanced styling prompt
        enhanced_styling = "\n\n".join(styling_parts)
        
        return enhanced_styling
    
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
        Generate an image of apparel fitted on a model using hybrid approach:
        1. Extract apparel features with OpenAI
        2. Generate try-on with Gemini using enhanced prompt
        
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
            # Step 1: Extract detailed features from apparel using OpenAI
            apparel_features = self.extract_apparel_features_with_openai(apparel_image)
            
            # Step 2: Create an enhanced styling prompt based on extracted features
            enhanced_styling = self.create_enhanced_styling_prompt(
                apparel_features,
                base_styling=styling_prompt,
                apparel_color=apparel_color
            )
            
            # Log the enhanced styling prompt
            print(f"Enhanced styling prompt: {enhanced_styling}")
            
            # Step 3: Use Gemini to generate the try-on with the enhanced prompt
            final_image, error, high_quality_fitted = self.imagen_handler.fit_apparel_on_model(
                apparel_image,
                model_attributes,
                pose,
                model_image=model_image,
                photoshoot_settings=photoshoot_settings,
                styling_prompt=enhanced_styling,  # Use the enhanced prompt here
                apparel_color=apparel_color if apparel_color else apparel_features.get('color_hex_exact')
            )
            
            if final_image:
                # Create a composite image with information about the hybrid approach
                width, height = 800, 1100  # Taller to accommodate feature info
                composite = Image.new('RGB', (width, height), color=(245, 245, 250))
                
                # Resize images for the composite
                apparel_resized = apparel_image.copy()
                apparel_resized.thumbnail((width//3 - 20, height//5 - 20))
                
                if model_image:
                    model_resized = model_image.copy()
                    model_resized.thumbnail((width//3 - 20, height//5 - 20))
                
                fitted_resized = high_quality_fitted.copy() if high_quality_fitted else final_image.copy()
                fitted_resized.thumbnail((width - 40, height//2))
                
                # Paste images into composite
                composite.paste(apparel_resized, (10, 60))
                
                if model_image:
                    composite.paste(model_resized, (width//3 + 10, 60))
                
                # Center the fitted image in the middle section
                fit_x = (width - fitted_resized.width) // 2
                composite.paste(fitted_resized, (fit_x, height//5 + 80))
                
                # Add labels and information using PIL's ImageDraw
                draw = ImageDraw.Draw(composite)
                
                # Try to use a nice font, fall back to default if not available
                try:
                    font = ImageFont.truetype("Arial", 16)
                    title_font = ImageFont.truetype("Arial", 20)
                    small_font = ImageFont.truetype("Arial", 12)
                except IOError:
                    font = ImageFont.load_default()
                    title_font = font
                    small_font = font
                
                # Add captions
                caption = f"Virtual Photoshoot - {pose} Pose - Enhanced Color Matching"
                draw.text((width//2 - 220, 20), caption, fill=(0, 0, 0), font=title_font)
                draw.text((10, 40), "Original Apparel", fill=(0, 0, 0), font=font)
                
                if model_image:
                    draw.text((width//3 + 10, 40), "Your Model", fill=(0, 0, 0), font=font)
                
                draw.text((width//2 - 100, height//5 + 60), "Final Result (Enhanced)", fill=(0, 0, 0), font=font)
                
                # Add feature information with color swatch
                features_y = height//5 + fitted_resized.height + 100
                draw.text((10, features_y), "Extracted Apparel Features:", fill=(0, 0, 0), font=font)
                
                # Add color swatch
                color_hex = apparel_color if apparel_color else apparel_features.get('color_hex_exact', "#000000")
                color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Draw color swatch
                swatch_x, swatch_y = 10, features_y + 30
                swatch_size = 40
                draw.rectangle([(swatch_x, swatch_y), (swatch_x + swatch_size, swatch_y + swatch_size)], fill=color_rgb, outline=(0, 0, 0))
                
                # Add color info next to swatch
                draw.text((swatch_x + swatch_size + 10, swatch_y), f"Exact Color: {color_hex}", fill=(0, 0, 0), font=font)
                draw.text((swatch_x + swatch_size + 10, swatch_y + 20), f"Description: {apparel_features.get('dominant_color', 'Unknown')}", fill=(0, 0, 0), font=small_font)
                
                # Add key features in a readable format
                pattern_desc = apparel_features.get('pattern_description', apparel_features.get('pattern_type', 'solid'))
                texture_desc = apparel_features.get('texture_description', 'standard texture')
                
                feature_text = f"Type: {apparel_features['apparel_type']} | " + \
                               f"Pattern: {pattern_desc} | " + \
                               f"Texture: {texture_desc[:30]}..."
                
                draw.text((10, features_y + 80), feature_text, fill=(0, 0, 0), font=small_font)
                
                return composite, "Successfully fitted apparel on model with precise color matching", high_quality_fitted
            else:
                return None, f"Error in hybrid approach: {error}", None
            
        except Exception as e:
            return None, f"Error in hybrid approach: {str(e)}", None