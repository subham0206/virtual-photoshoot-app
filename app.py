import streamlit as st
import os
import io
from PIL import Image
import time
from utils import ImagenHandler, extract_apparel_features
from openai_handler import OpenAIDALLE3Handler
from hybrid_generation import HybridApparelFittingHandler
import socket
import sys

# Set the API keys
try:
    # Try to get API key from Streamlit secrets (for deployment)
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback to hardcoded key for local development
    google_api_key = ""
    openai_api_key = ""

# Initialize handlers for all approaches
imagen_handler = ImagenHandler(google_api_key, timeout=90, 
                              imagen_model="models/imagen-4.0-generate-001",
                              gemini_image_model="gemini-2.5-flash-image-preview")

# Initialize OpenAI DALL-E 3 handler
dalle3_handler = OpenAIDALLE3Handler(openai_api_key, timeout=90, model="dall-e-3")

# Initialize hybrid approach handler
hybrid_handler = HybridApparelFittingHandler(openai_api_key, google_api_key, timeout=90)

# Set page configuration
st.set_page_config(
    page_title="B+C Virtual Photoshoot App",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add some custom CSS to handle network issues with font loading
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .network-warning {
        padding: 10px;
        background-color: #FFEB3B;
        border-radius: 5px;
        margin-bottom: 10px;
        display: none;
    }
</style>
<div class="network-warning" id="network-warning">
    Network connection issues detected. Some features may be slow to respond.
</div>
<script>
    // Simple script to detect network issues
    window.addEventListener('offline', function() {
        document.getElementById('network-warning').style.display = 'block';
    });
    window.addEventListener('online', function() {
        document.getElementById('network-warning').style.display = 'none';
    });
    
    // Check for slow connections
    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    if (connection && connection.effectiveType && (connection.effectiveType === '2g' || connection.effectiveType === 'slow-2g')) {
        document.getElementById('network-warning').style.display = 'block';
    }
</script>
""", unsafe_allow_html=True)

# App title and description
st.title("B+C Virtual Photoshoot App")
st.markdown("""
This application allows you to upload apparel and fit it onto an AI-generated model.
The app preserves the color, texture, and size of your apparel while placing it on a model with your chosen attributes.
""")

# Predefined models data
PREDEFINED_MODELS = {
    "Male": [
        {
            "id": "m1",
            "name": "African American – Tall Athletic",
            "description": "Tall athletic build, fade haircut with curly top",
            "image_path": "models/male_african_american_tall.png",
            "attributes": {
                "gender": "Male",
                "ethnicity": "African American",
                "skin_tone": "Dark skin",
                "build": "Tall athletic",
                "hair_color": "Black",
                "hair_style": "Fade with curly top",
                "facial_hair": "Light stubble"
            }
        },
        {
            "id": "m2",
            "name": "Spanish",
            "description": "Tan skin, slim build, brown buzz cut",
            "image_path": "models/male_spanish.png",
            "attributes": {
                "gender": "Male",
                "ethnicity": "Spanish",
                "skin_tone": "Tan skin",
                "build": "Slim", # Changed from Athletic to Slim
                "hair_color": "Brown",
                "hair_style": "Buzz cut",
                "facial_hair": "None"
            }
        },
        {
            "id": "m3",
            "name": "Japanese",
            "description": "Athletic build, dark brown buzz cut",
            "image_path": "models/male_japanese.png",
            "attributes": {
                "gender": "Male",
                "ethnicity": "Japanese",
                "skin_tone": "Light skin",
                "build": "Athletic",
                "hair_color": "Dark brown",
                "hair_style": "Buzz cut",
                "facial_hair": "None"
            }
        },
        {
            "id": "m4",
            "name": "Middle Eastern",
            "description": "Tall slender build, buzz cut, light stubble",
            "image_path": "models/male_middle_eastern.png",
            "attributes": {
                "gender": "Male",
                "ethnicity": "Middle Eastern",
                "skin_tone": "Medium skin",
                "build": "Tall slender",
                "hair_color": "Black",
                "hair_style": "Buzz cut", # Changed from wavy hair to buzz cut
                "facial_hair": "Light stubble" # Changed from beard to stubble
            }
        },
        {
            "id": "m5",
            "name": "Mixed Race",
            "description": "Tall athletic build, buzz cut, light stubble",
            "image_path": "models/male_mixed_race.png",
            "attributes": {
                "gender": "Male",
                "ethnicity": "Mixed Race (Black/Caucasian)",
                "skin_tone": "Medium skin",
                "build": "Tall athletic",
                "hair_color": "Dark brown",
                "hair_style": "Buzz cut", # Changed from curly hair to buzz cut
                "facial_hair": "Light stubble"
            }
        }
    ],
    "Female": [
        {
            "id": "f1",
            "name": "African American – Tall",
            "description": "Tall slender build, box braids",
            "image_path": "models/female_african_american_tall.png",
            "attributes": {
                "gender": "Female",
                "ethnicity": "African American",
                "skin_tone": "Dark skin",
                "build": "Tall slender",
                "hair_color": "Black",
                "hair_style": "Box braids" # Changed from long natural curly to box braids
            }
        },
        {
            "id": "f2",
            "name": "Middle Eastern",
            "description": "Tall slender build, long straight dark brown hair",
            "image_path": "models/female_middle_eastern.png",
            "attributes": {
                "gender": "Female",
                "ethnicity": "Middle Eastern",
                "skin_tone": "Medium skin",
                "build": "Tall slender",
                "hair_color": "Dark brown",
                "hair_style": "Long straight" # Changed from wavy windblown to straight
            }
        },
        {
            "id": "f3",
            "name": "Mixed Race",
            "description": "Tall slender build, medium-length curly blonde hair",
            "image_path": "models/female_mixed_race.png",
            "attributes": {
                "gender": "Female",
                "ethnicity": "Mixed Race (Asian/Caucasian)",
                "skin_tone": "Light medium skin",
                "build": "Tall slender",
                "hair_color": "Blonde", # Changed from honey-brown to blonde
                "hair_style": "Medium-length curly"
            }
        },
        {
            "id": "f4",
            "name": "South Asian",
            "description": "Tall slender build, shoulder-length wavy bob",
            "image_path": "models/female_south_asian.png",
            "attributes": {
                "gender": "Female",
                "ethnicity": "South Asian",
                "skin_tone": "Medium skin",
                "build": "Tall slender",
                "hair_color": "Black",
                "hair_style": "Shoulder-length wavy bob" # Changed from long straight to wavy bob
            }
        },
        {
            "id": "f5",
            "name": "Native American",
            "description": "Thin build, above sleeve length black hair",
            "image_path": "models/female_native_american.png",
            "attributes": {
                "gender": "Female",
                "ethnicity": "Native American",
                "skin_tone": "Tan skin",
                "build": "Thin",
                "hair_color": "Black",
                "hair_style": "Above sleeve length" # Changed from long to above sleeve length
            }
        }
    ]
}

# Bella+Canvas color swatches - hex color codes
BELLA_CANVAS_COLORS = {
    "Black": "#000000",
    "White": "#FFFFFF",
    "Ash Gray": "#CDD0CD",
    "Athletic Heather": "#D0D0D0",
    "Dark Gray Heather": "#686868",
    "Deep Heather": "#7C7C7C",
    "Heather Navy": "#373F51",
    "Heather Forest": "#2A4734",
    "Kelly Green": "#4CAF50",
    "Forest Green": "#1B4D3E",
    "Olive": "#808000",
    "Navy": "#000080",
    "Royal Blue": "#4169E1",
    "True Royal": "#2D68C4",
    "Team Purple": "#3F2A7E",
    "Purple": "#800080",
    "Mauve": "#E0B0FF",
    "Red": "#FF0000",
    "Cardinal": "#C41E3A",
    "Brick Red": "#CB4154",
    "Coral": "#FF7F50",
    "Orange": "#FFA500",
    "Gold": "#FFD700",
    "Yellow": "#FFFF00",
    "Army": "#4B5320",
    "Teal": "#008080",
    "Aqua": "#00FFFF",
    "Pink": "#FFC0CB",
    "Soft Pink": "#FFCCCB",
    "Maroon": "#800000",
    "Brown": "#A52A2A",
    "Tan": "#D2B48C"
}

# Create sidebar for app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Apparel", "Select Model", "Photoshoot"])

# Global session state to store uploaded image and generated model
if "apparel_image" not in st.session_state:
    st.session_state.apparel_image = None
if "model_image" not in st.session_state:
    st.session_state.model_image = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "final_image" not in st.session_state:
    st.session_state.final_image = None
if "model_attributes" not in st.session_state:
    st.session_state.model_attributes = {}
if "photoshoot_variations" not in st.session_state:
    st.session_state.photoshoot_variations = []
if "network_retry_count" not in st.session_state:
    st.session_state.network_retry_count = 0
    
# Add session state variables for apparel attributes
if "clothing_type" not in st.session_state:
    st.session_state.clothing_type = "N/A"
if "clothing_fit_style" not in st.session_state:
    st.session_state.clothing_fit_style = "N/A"
if "clothing_color" not in st.session_state:
    st.session_state.clothing_color = "N/A"
if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = ""

# Display network status and advice in the sidebar
with st.sidebar.expander("Network Status", expanded=False):
    st.info("""
    If you experience connection issues:
    1. Try refreshing the page
    2. Check your network connection
    3. Reduce image sizes before uploading
    """)

# Upload Apparel page
if page == "Upload Apparel":
    st.header("Upload Apparel Image")
    uploaded_file = st.file_uploader("Choose an apparel image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.apparel_image = image
        
        # Add the three detailed clothing information sections
        st.subheader("Clothing Information")
        st.info("The information you provide below will be used to generate the final photoshoot image. All fields default to N/A.")
        
        # 1. Clothing Type Section
        with st.expander("1. Clothing Type", expanded=True):
            # Add a dedicated option for selecting B+C product types first
            use_bc_products = st.checkbox("Use Bella+Canvas Product Types", value=False)
            
            if use_bc_products:
                # B+C specific product types
                clothing_type = st.selectbox(
                    "Select Bella+Canvas Product Type",
                    options=[
                        "N/A",
                        "Women's Cotton Spandex Legging",
                        "Infant Jersey Short Sleeve One Piece",
                        "Women's Micro Rib Baby Tee",
                        "Women's Micro Rib Spaghetti Strap Tank",
                        "Women's Micro Rib Muscle Crop Tank",
                        "Women's Micro Rib Racer Tank",
                        "Women's Baby Rib Tank",
                        "Women's Micro Ribbed Tank",
                        "Infant Triblend Short Sleeve One Piece",
                        "Women's Micro Rib 3/4 Raglan Baby Tee",
                        "Women's Micro Rib Raglan Baby Tee",
                        "Women's Micro Rib Long Sleeve Baby Tee",
                        "Unisex Jersey Short Sleeve Tee",
                        "Infant Jersey Short Sleeve Tee",
                        "Unisex Heather CVC Short Sleeve Tee",
                        "Unisex EcoMax Short Sleeve Tee",
                        "Toddler Jersey Short Sleeve Tee",
                        "Unisex Made in the USA Jersey Short Sleeve Tee",
                        "Youth Jersey Short Sleeve Tee",
                        "Youth Heather CVC Short Sleeve Tee",
                        "Mens Jersey Short Sleeve Tee With Curved Hem",
                        "Unisex Jersey Short Sleeve V-Neck Tee",
                        "Unisex Heather CVC Short Sleeve V-Neck Tee",
                        "Men's Long Body Urban Tee",
                        "Unisex 6 oz Heavyweight Tee",
                        "Youth 6 oz Heavyweight Tee",
                        "Men's Jersey Short Sleeve Pocket Tee",
                        "Unisex 3/4 Sleeve Baseball Tee",
                        "Toddler 3/4 Sleeve Baseball Tee",
                        "Youth 3/4 Sleeve Baseball Tee",
                        "Men's Heather CVC Raglan Tee",
                        "Unisex Sueded Short Sleeve Tee",
                        "Unisex Triblend Short Sleeve Tee",
                        "Infant Triblend Short Sleeve Tee",
                        "Toddler Triblend Short Sleeve Tee",
                        "Youth Triblend Short Sleeve Tee",
                        "Unisex Triblend Short Sleeve V-Neck Tee",
                        "Unisex Jersey Tank",
                        "Unisex Heather CVC Tank",
                        "Youth Jersey Tank",
                        "Youth Heather CVC Tank",
                        "Unisex Jersey Muscle Tank",
                        "Unisex Triblend Tank",
                        "Unisex Jersey Long Sleeve Tee",
                        "Unisex Heather CVC Long Sleeve Tee",
                        "Youth Jersey Long Sleeve Tee",
                        "Youth Heather CVC Long Sleeve Tee",
                        "Toddler Jersey Long Sleeve Tee",
                        "Unisex 6 oz Heavyweight Long Sleeve Tee",
                        "Youth 6 oz Heavyweight Long Sleeve Tee",
                        "Unisex Jersey Long Sleeve Hoodie",
                        "Unisex Triblend Long Sleeve Tee",
                        "Youth Triblend Long Sleeve Tee",
                        "Unisex Poly-Cotton Short Sleeve Tee",
                        "Unisex V-Neck Textured Tee",
                        "Unisex Sponge Fleece Pullover Hoodie",
                        "Toddler Sponge Fleece Pullover Hoodie",
                        "Youth Sponge Fleece Pullover Hoodie",
                        "Unisex Sponge Fleece Sweatshort",
                        "Unisex Sponge Fleece Straight Leg Sweatpant",
                        "Unisex Sponge Fleece Jogger Sweatpants",
                        "Toddler Sponge Fleece Jogger Sweatpants",
                        "Youth Sponge Fleece Jogger Sweatpants",
                        "Unisex Sponge Fleece DTM Hoodie",
                        "Unisex Sponge Fleece Long Scrunch Pant",
                        "Unisex Sponge Fleece Full-Zip Hoodie",
                        "Toddler Sponge Fleece Full- Zip Hoodie",
                        "Youth Sponge Fleece Full-Zip Hoodie",
                        "Unisex Sponge Fleece DTM Full-Zip Hoodie",
                        "Women's Cutoff Sweatshort",
                        "Unisex Sponge Fleece Raglan Sweatshirt",
                        "Toddler Sponge Fleece Raglan Sweatshirt",
                        "Youth Sponge Fleece Raglan Sweatshirt",
                        "Unisex Triblend Fleece Zip Hoodie",
                        "Unisex Sponge Fleece Classic Crewneck Sweatshirt",
                        "Unisex Triblend Full-Zip Lightweight Hoodie",
                        "Unisex Sponge Fleece Drop Shoulder Sweatshirt",
                        "Unisex 7.5 oz Heavyweight Tee",
                        "Unisex 7.5 oz Heavyweight Long Sleeve Tee",
                        "Unisex 10 oz Heavyweight Crewneck Sweatshirt",
                        "Unisex 10 oz Heavyweight Pullover Hoodie",
                        "Unisex 10 oz Heavyweight Sweatpant",
                        "Unisex Heavyweight Garment Dye Tee",
                        "Unisex Heavyweight Garment Dye Long Sleeve Tee",
                        "Women's Jersey Muscle Tank",
                        "Women's Slim Fit Tee",
                        "Women's Jersey Short Sleeve V-Neck Tee",
                        "Women's Jersey Racerback Tank",
                        "Women's 6 oz Heavyweight Tee",
                        "Women's Relaxed Jersey Short Sleeve Tee",
                        "Women's Relaxed Heather CVC Short Sleeve Tee",
                        "Women's Relaxed Jersey Short Sleeve V-Neck Tee",
                        "Women's Relaxed Heather CVC Short Sleeve V-Neck Tee",
                        "Women's Relaxed Triblend Short Sleeve Tee",
                        "Women's Relaxed Triblend Short Sleeve V-Neck Tee",
                        "Womens Jersey Crop Tee",
                        "Women's Jersey Long Sleeve Tee",
                        "Women's Cropped Long Sleeve Tee",
                        "Women's Poly-Cotton Crop Tee",
                        "Women's Racerback Cropped Tank",
                        "Women's Cropped Fleece Hoodie",
                        "Women's Cropped Crew Fleece",
                        "Women's Raglan Pullover Fleece",
                        "Women's Crewneck Pullover",
                        "Women's Classic Hooded Pullover",
                        "Women's Triblend Short Sleeve Tee",
                        "Women's Triblend Racerback Tank",
                        "Women's Cropped Long Sleeve Hoodie",
                        "Women's Flowy Racerback Tank",
                        "Youth Flowy Racerback Tank",
                        "Womens Flowy Scoop Muscle Tank",
                        "Women's Flowy Muscle Tee with Rolled Cuff",
                        "Women's Slouchy V-Neck Tee",
                        "Women's Slouchy Tee",
                        "Women's Slouchy Tank",
                        "Women's Flowy Cropped Tee"
                    ],
                    help="Select the specific Bella+Canvas product type"
                )
                
                st.info("You've selected a Bella+Canvas product type. This will be used in the photoshoot prompt.")
                
            else:
                # Show regular clothing category options
                clothing_category = st.selectbox(
                    "Clothing Category",
                    options=["N/A", "Tops", "Bottoms", "Footwear", "Accessories"],
                    help="Select the category of clothing item"
                )
                
                # Show relevant options based on the selected category
                if clothing_category == "Tops":
                    clothing_type = st.selectbox(
                        "Top Type",
                        options=["N/A", "T-shirt", "Blouse", "Hoodie", "Jacket", "Sweater", "Tank top", "Other"]
                    )
                    if clothing_type == "Other":
                        custom_type = st.text_input("Specify Top Type")
                        clothing_type = custom_type if custom_type else "Custom top"
                    
                elif clothing_category == "Bottoms":
                    clothing_type = st.selectbox(
                        "Bottom Type",
                        options=["N/A", "Jeans", "Trousers", "Shorts", "Skirts", "Leggings", "Joggers", "Other"]
                    )
                    if clothing_type == "Other":
                        custom_type = st.text_input("Specify Bottom Type")
                        clothing_type = custom_type if custom_type else "Custom bottoms"
                        
                elif clothing_category == "Footwear":
                    clothing_type = st.selectbox(
                        "Footwear Type",
                        options=["N/A", "Sneakers", "Boots", "Sandals", "Loafers", "Athletic shoes", "Heels", "Other"]
                    )
                    if clothing_type == "Other":
                        custom_type = st.text_input("Specify Footwear Type")
                        clothing_type = custom_type if custom_type else "Custom footwear"
                        
                elif clothing_category == "Accessories":
                    clothing_type = st.selectbox(
                        "Accessory Type",
                        options=["N/A", "Belt", "Watch", "Jewelry", "Hat", "Bag", "Other"]
                    )
                    if clothing_type == "Other":
                        custom_type = st.text_input("Specify Accessory Type")
                        clothing_type = custom_type if custom_type else "Custom accessory"
                else:
                    clothing_type = "N/A"
            
            # Custom description field for additional details
            custom_description = st.text_area(
                "Additional Description (Optional)",
                placeholder="Add any additional details about the clothing item"
            )
            
            # Update session state
            if clothing_type != "N/A" or custom_description:
                full_type = f"{clothing_type}"
                if custom_description:
                    full_type += f" ({custom_description})"
                st.session_state.clothing_type = full_type
            else:
                st.session_state.clothing_type = "N/A"
        
        # 2. Clothing Fit and Style Section
        with st.expander("2. Styling(uploaded apparel)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                fit_type = st.selectbox(
                    "Fit",
                    options=["N/A", "Regular fit", "Oversized", "Loose fit", "Relaxed fit", "Slim fit", "Tight fit", "Other"]
                )
                if fit_type == "Other":
                    custom_fit = st.text_input("Specify Fit")
                    fit_type = custom_fit if custom_fit else "Custom fit"
                
                style_type = st.selectbox(
                    "Style",
                    options=["N/A", "Casual", "Athletic", "Streetwear", "Formal", "Business casual", "Other"]
                )
                if style_type == "Other":
                    custom_style = st.text_input("Specify Style")
                    style_type = custom_style if custom_style else "Custom style"
            
            with col2:
                length_type = st.selectbox(
                    "Length",
                    options=["N/A", "Crop", "Knee-length", "Ankle-length", "Full-length", "Short", "Other"]
                )
                if length_type == "Other":
                    custom_length = st.text_input("Specify Length")
                    length_type = custom_length if custom_length else "Custom length"
                
                pattern_type = st.selectbox(
                    "Pattern",
                    options=["N/A", "Solid", "Striped", "Plaid", "Checkered", "Geometric", "Floral", "Other"]
                )
                if pattern_type == "Other":
                    custom_pattern = st.text_input("Specify Pattern")
                    pattern_type = custom_pattern if custom_pattern else "Custom pattern"
            
            fabric_type = st.selectbox(
                "Fabric",
                options=["N/A", "Denim", "Cotton", "Leather", "Silk", "Wool", "Linen", "Polyester", "Other"]
            )
            if fabric_type == "Other":
                custom_fabric = st.text_input("Specify Fabric")
                fabric_type = custom_fabric if custom_fabric else "Custom fabric"
            
            # Custom description field for additional fit/style details
            fit_style_description = st.text_area(
                "Additional Fit/Style Details (Optional)",
                placeholder="Add any additional details about the fit or style"
            )
            
            # Update session state
            fit_style_parts = []
            if fit_type != "N/A":
                fit_style_parts.append(fit_type)
            if style_type != "N/A":
                fit_style_parts.append(style_type)
            if length_type != "N/A":
                fit_style_parts.append(length_type)
            if pattern_type != "N/A":
                fit_style_parts.append(pattern_type)
            if fabric_type != "N/A":
                fit_style_parts.append(f"{fabric_type} fabric")
                
            if fit_style_parts:
                full_fit_style = ", ".join(fit_style_parts)
                if fit_style_description:
                    full_fit_style += f" ({fit_style_description})"
                st.session_state.clothing_fit_style = full_fit_style
            elif fit_style_description:
                st.session_state.clothing_fit_style = fit_style_description
            else:
                st.session_state.clothing_fit_style = "N/A"
        
        # 3. Clothing Color Section
        with st.expander("3. Clothing Color", expanded=True):
            # The existing color change option can remain, but we'll add more specific color selections
            st.info("Select colors for your apparel item. This will be used along with or instead of the uploaded image colors.")
            
            # Initialize default color options
            color_options = [
                "N/A", "White", "Black", "Red", "Blue", "Green", "Yellow", "Purple", "Pink", 
                "Navy", "Gray", "Striped", "Patterned", "Heathered", "Other"
            ]
            
            # Update color options based on clothing category or product type
            if not use_bc_products and 'clothing_category' in locals():
                if clothing_category == "Bottoms":
                    color_options = [
                        "N/A", "Light blue jeans", "Dark wash jeans", "Medium wash jeans", "Black jeans",
                        "Black leggings", "Gray trousers", "Khaki", "Navy", "Other"
                    ]
                elif clothing_category == "Footwear":
                    color_options = [
                        "N/A", "White sneakers", "Black boots", "Tan sandals", "Silver sneakers", 
                        "Brown leather", "Red", "Multi-color", "Other"
                    ]
                elif clothing_category == "Accessories":
                    color_options = [
                        "N/A", "Gold", "Silver", "Black", "Brown", "White", "Colorful", "Other"
                    ]
            
            color_type = st.selectbox("Color Description", options=color_options)
            
            if color_type == "Other":
                custom_color = st.text_input("Specify Color Description")
                color_type = custom_color if custom_color else "Custom color"
            
            # Show additional color details option
            color_details = st.text_area(
                "Additional Color Details (Optional)",
                placeholder="Describe any color details, patterns, gradients, etc."
            )
            
            # Update session state
            if color_type != "N/A":
                if color_details:
                    st.session_state.clothing_color = f"{color_type} ({color_details})"
                else:
                    st.session_state.clothing_color = color_type
            elif color_details:
                st.session_state.clothing_color = color_details
            else:
                st.session_state.clothing_color = "N/A"
            
            # Note about color change option
            if "apparel_color" in st.session_state and st.session_state.apparel_color:
                st.info("Note: You also have a specific color change applied from the 'Change Apparel Color' section.")
        
        # Add color change option (existing code)
        with st.expander("Change Apparel Color", expanded=False):
            st.info("Use this option if you want to change the color of your apparel.")
            color_change = st.checkbox("I want to change the apparel color")
            
            if color_change:
                # Color selection options using Bella+Canvas color swatches
                color_method = st.radio("Color Selection Method", ["Bella+Canvas Colors", "Custom Color"])
                
                if color_method == "Bella+Canvas Colors":
                    # Display Bella+Canvas color options
                    color_name = st.selectbox("Select Color", list(BELLA_CANVAS_COLORS.keys()))
                    selected_color = BELLA_CANVAS_COLORS[color_name]
                    # Display the selected color
                    st.markdown(f"<div style='background-color:{selected_color}; width:50px; height:50px; border:1px solid black'></div>", unsafe_allow_html=True)
                    st.write(f"Selected color: {color_name} ({selected_color})")
                    st.session_state.apparel_color = selected_color
                    st.session_state.apparel_color_name = color_name
                else:
                    # Custom color options
                    custom_color_method = st.radio("Custom Color Method", ["Color Picker", "Hex Code Input"])
                    
                    if custom_color_method == "Color Picker":
                        selected_color = st.color_picker("Select New Color", "#0000FF")
                        st.session_state.apparel_color = selected_color
                        st.session_state.apparel_color_name = "Custom"
                    else:
                        selected_color = st.text_input("Enter Hex Color Code", "#0000FF")
                        # Validate hex code and ensure it starts with #
                        if selected_color:
                            if not selected_color.startswith('#'):
                                selected_color = '#' + selected_color
                            st.session_state.apparel_color = selected_color
                            st.session_state.apparel_color_name = "Custom"
                
                # Display the selected color
                if "apparel_color" in st.session_state and st.session_state.apparel_color:
                    st.markdown(f"<div style='background-color:{st.session_state.apparel_color}; width:50px; height:50px; border:1px solid black'></div>", unsafe_allow_html=True)
                    st.write(f"Selected color: {st.session_state.apparel_color}")
                
                # Apply color button
                if st.button("Apply Color"):
                    with st.spinner("Applying color..."):
                        try:
                            # Verify we have a valid color before proceeding
                            if st.session_state.apparel_color:
                                st.success(f"Color {st.session_state.apparel_color} will be applied during photoshoot generation.")
                            else:
                                st.error("Please select a valid color first.")
                        except Exception as e:
                            st.error(f"Error applying color: {str(e)}")
            else:
                # Reset the color if checkbox is unchecked
                st.session_state.apparel_color = None
                st.session_state.apparel_color_name = None
        
        # Display the current selections and generate a preview of the final prompt
        st.subheader("Current Apparel Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Apparel Image", width=250)
            
        with col2:
            if st.session_state.clothing_type != "N/A":
                st.markdown(f"**Clothing Type:** {st.session_state.clothing_type}")
            else:
                st.markdown("**Clothing Type:** N/A")
                
            if st.session_state.clothing_fit_style != "N/A":
                st.markdown(f"**Fit & Style:** {st.session_state.clothing_fit_style}")
            else:
                st.markdown("**Fit & Style:** N/A")
                
            if st.session_state.clothing_color != "N/A":
                st.markdown(f"**Color Description:** {st.session_state.clothing_color}")
            else:
                st.markdown("**Color Description:** N/A")
                
            if "apparel_color" in st.session_state and st.session_state.apparel_color:
                color_name = st.session_state.apparel_color_name if "apparel_color_name" in st.session_state else "Custom"
                st.markdown(f"**Color Change:** {color_name}")
                st.markdown(f"<div style='background-color:{st.session_state.apparel_color}; width:50px; height:50px; border:1px solid black'></div>", unsafe_allow_html=True)
        
        # Generate and display preview of the final prompt
        st.subheader("Final Prompt Preview")
        prompt_parts = []
        
        if st.session_state.clothing_type != "N/A":
            prompt_parts.append(f"Clothing Type: {st.session_state.clothing_type}")
        
        if st.session_state.clothing_fit_style != "N/A":
            prompt_parts.append(f"Fit and Style: {st.session_state.clothing_fit_style}")
        
        if st.session_state.clothing_color != "N/A":
            prompt_parts.append(f"Color: {st.session_state.clothing_color}")
        
        if "apparel_color" in st.session_state and st.session_state.apparel_color:
            color_name = st.session_state.apparel_color_name if "apparel_color_name" in st.session_state else "Custom"
            prompt_parts.append(f"Apply color change to: {color_name} ({st.session_state.apparel_color})")
        
        if prompt_parts:
            final_prompt = "The model should be wearing apparel with these specifications:\n" + "\n".join(prompt_parts)
            st.session_state.final_prompt = final_prompt
            st.info(final_prompt)
        else:
            st.session_state.final_prompt = ""
            st.info("No specific apparel details have been provided. The model will wear the uploaded item as shown.")
        
        st.success("Apparel image uploaded successfully! Proceed to 'Select Model' to choose a model for your apparel.")

# Select Model page (renamed from Generate Model)
elif page == "Select Model":
    st.header("Select Model")
    
    if st.session_state.apparel_image is None:
        st.warning("Please upload an apparel image first.")
    else:
        # Display the uploaded apparel
        st.subheader("Uploaded Apparel")
        st.image(st.session_state.apparel_image, width=300)
        
        # Gender selection tabs for model selection
        gender_tab = st.radio("Select Gender", ["Male", "Female"])
        
        # Display model selection based on gender
        st.subheader(f"Select {gender_tab} Model")
        
        # Create columns for model display
        cols = st.columns(min(5, len(PREDEFINED_MODELS[gender_tab])))
        
        for i, model in enumerate(PREDEFINED_MODELS[gender_tab]):
            with cols[i % len(cols)]:
                # Check if model image exists, if not display placeholder
                try:
                    if os.path.exists(model["image_path"]):
                        model_img = Image.open(model["image_path"])
                    else:
                        # If image doesn't exist yet, generate a placeholder
                        # This is temporary - in production, these would be pre-generated
                        st.info(f"Using placeholder for {model['name']}")
                        model_img = None
                except:
                    model_img = None
                
                # Display model image or placeholder
                if model_img:
                    st.image(model_img, caption=model["name"], use_column_width=True)
                else:
                    st.markdown(f"### {model['name']}")
                    st.markdown(model["description"])
                
                # Selection button for this model
                if st.button(f"Select {model['name']}", key=f"select_{model['id']}"):
                    st.session_state.selected_model = model
                    st.session_state.model_attributes = model["attributes"]
                    
                    # If model image exists, use it
                    if model_img:
                        st.session_state.model_image = model_img
                    else:
                        # In a real implementation, you would have these pre-generated
                        # For now, we'll just note that this would normally generate the model
                        st.info(f"Model {model['name']} selected. In production, this would use a pre-generated image.")
                        st.session_state.model_image = None
        
        # Display selected model if any
        if st.session_state.selected_model:
            st.success(f"You have selected: {st.session_state.selected_model['name']}")
            st.info(f"Description: {st.session_state.selected_model['description']}")
            
            # Add styling input for the selected model
            st.subheader("Styling Input")
            
            # Initialize styling prompt in session state if not already present
            if "styling_prompt" not in st.session_state:
                st.session_state.styling_prompt = ""
            
            styling_prompt = st.text_area(
                "Styling Instructions", 
                value=st.session_state.styling_prompt,
                placeholder="Example: tucked in, oversized fit, rolled sleeves, textured fabric, etc.",
                help="Add specific styling instructions for how the garment should appear on the model."
            )
            
            st.session_state.styling_prompt = styling_prompt
            
            if styling_prompt:
                st.success("Styling instructions will be applied during photoshoot generation.")

elif page == "Photoshoot":
    st.header("Virtual Photoshoot")
    
    if st.session_state.apparel_image is None:
        st.warning("Please upload an apparel image first.")
    elif st.session_state.selected_model is None:
        st.warning("Please select a model first.")
    else:
        # Display the uploaded apparel and selected model
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Apparel")
            st.image(st.session_state.apparel_image, width=300)
            
            # Display detailed apparel information
            with st.expander("Apparel Information", expanded=True):
                if st.session_state.clothing_type != "N/A":
                    st.markdown(f"**Clothing Type:** {st.session_state.clothing_type}")
                
                if st.session_state.clothing_fit_style != "N/A":
                    st.markdown(f"**Fit & Style:** {st.session_state.clothing_fit_style}")
                
                if st.session_state.clothing_color != "N/A":
                    st.markdown(f"**Color Description:** {st.session_state.clothing_color}")
                    
                # Display color information if a color was selected
                if "apparel_color" in st.session_state and st.session_state.apparel_color:
                    color_name = st.session_state.apparel_color_name if "apparel_color_name" in st.session_state else "Custom"
                    st.markdown(f"**Selected Color:** {color_name}")
                    st.markdown(f"<div style='background-color:{st.session_state.apparel_color}; width:50px; height:50px; border:1px solid black'></div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Your Model")
            if st.session_state.model_image:
                st.image(st.session_state.model_image, width=300)
            else:
                # Display model information as text if image isn't available
                st.info(f"Model: {st.session_state.selected_model['name']}")
                st.info(f"Description: {st.session_state.selected_model['description']}")
        
        # Create tabs for different photoshoot configuration options
        view_tab, background_tab, lighting_tab = st.tabs(["View Angle", "Background", "Lighting"])
        
        # View angle tab
        with view_tab:
            st.subheader("Select View Angle")
            view_options = [
                "Front view", "Back view", "Side view (left)", "Side view (right)", 
                "Three-quarter front", "Three-quarter back"
            ]
            
            view_angle = st.selectbox("Choose a view angle", options=view_options, index=0)
        
        # Background tab
        with background_tab:
            st.subheader("Select Background")
            background_options = [
                "Light grey studio cyc, no shadows", 
                "White studio", 
                "Dark grey studio", 
                "Black studio",
                "Minimalist interior", 
                "Urban street", 
                "Nature outdoor", 
                "Solid color"
            ]
            
            background = st.selectbox("Choose a background", options=background_options, index=0)
            
            # Show additional options based on background selection
            if background == "Solid color":
                bg_color_name = st.selectbox("Choose background color", list(BELLA_CANVAS_COLORS.keys()))
                background_color = BELLA_CANVAS_COLORS[bg_color_name]
                st.markdown(f"<div style='background-color:{background_color}; width:100px; height:50px; border:1px solid black'></div>", unsafe_allow_html=True)
        
        # Lighting tab
        with lighting_tab:
            st.subheader("Lighting Settings")
            lighting_style = st.selectbox(
                "Lighting Style",
                options=[
                    "Standard studio", "Soft diffused", "Natural lighting", "Bright high-key"
                ],
                index=0
            )
        
        # Display styling prompt if provided
        if "styling_prompt" in st.session_state and st.session_state.styling_prompt:
            st.subheader("Styling Instructions")
            st.info(st.session_state.styling_prompt)
        
        # Store photoshoot settings in session state
        if "photoshoot_settings" not in st.session_state:
            st.session_state.photoshoot_settings = {}
        
        # Update photoshoot settings based on user selections
        st.session_state.photoshoot_settings = {
            "view_angle": view_angle,
            "background": background,
            "lighting_style": lighting_style
        }
        
        # Add background specific settings
        if background == "Solid color" and "background_color" in locals():
            st.session_state.photoshoot_settings["background_color"] = background_color
        
        # Show final prompt preview
        st.subheader("Final Generation Prompt")
        
        # Build the final prompt
        final_prompt_parts = ["The model should be wearing apparel with these specifications:"]
        
        # Add styling instructions first if provided (giving it higher priority)
        if "styling_prompt" in st.session_state and st.session_state.styling_prompt:
            final_prompt_parts.append(f"IMPORTANT STYLING INSTRUCTIONS: {st.session_state.styling_prompt}")
            final_prompt_parts.append("==== APPLY ABOVE STYLING INSTRUCTIONS WITH HIGHEST PRIORITY ====")
        
        # Add clothing type information
        if st.session_state.clothing_type != "N/A":
            final_prompt_parts.append(f"Clothing Type: {st.session_state.clothing_type}")
            
        # Add fit and style information
        if st.session_state.clothing_fit_style != "N/A":
            final_prompt_parts.append(f"Fit and Style: {st.session_state.clothing_fit_style}")
            
        # Add color description
        if st.session_state.clothing_color != "N/A":
            final_prompt_parts.append(f"Color: {st.session_state.clothing_color}")
            
        # Add color change if specified
        if "apparel_color" in st.session_state and st.session_state.apparel_color:
            color_name = st.session_state.apparel_color_name if "apparel_color_name" in st.session_state else "Custom"
            final_prompt_parts.append(f"Apply color change to: {color_name} ({st.session_state.apparel_color})")
            
        # Add photoshoot settings
        final_prompt_parts.append(f"View Angle: {view_angle}")
        final_prompt_parts.append(f"Background: {background}")
        if background == "Solid color" and "background_color" in locals():
            final_prompt_parts.append(f"Background Color: {bg_color_name} ({background_color})")
        final_prompt_parts.append(f"Lighting: {lighting_style}")
        
        # Display the final prompt
        final_prompt = "\n".join(final_prompt_parts)
        st.session_state.final_prompt = final_prompt
        
        with st.expander("View Complete Generation Prompt", expanded=True):
            st.info(final_prompt)
        
        # Fit apparel button
        # Add model selection
        st.subheader("AI Model Selection")
        ai_model = st.radio(
            "Select AI model for try-on generation",
            ["Hybrid (OpenAI + Gemini)", "OpenAI DALL-E 3", "Google Gemini"],
            index=0,  # Making hybrid the default option
            help="Hybrid approach uses OpenAI to extract apparel features and Gemini for the actual try-on, providing the best color accuracy and model consistency"
        )
        
        # Add explanation text based on selection
        if ai_model == "Hybrid (OpenAI + Gemini)":
            st.info("📝 **Hybrid Approach:** OpenAI extracts precise color, texture, and pattern details from your apparel, then Gemini generates the try-on while maintaining model consistency.")
        elif ai_model == "OpenAI DALL-E 3":
            st.warning("⚠️ **DALL-E 3:** May produce inaccurate colors and alter model appearance.")
        else:
            st.info("🔍 **Google Gemini:** Maintains model consistency but may not perfectly capture apparel colors and textures.")
        
        if st.button("Fit Apparel & Create Photoshoot"):
            with st.spinner("Creating your photoshoot... This may take a moment."):
                # Get styling prompt and combine with apparel details
                styling_prompt = st.session_state.styling_prompt if "styling_prompt" in st.session_state else ""
                
                # Add apparel details to the styling prompt for more accurate generation
                if st.session_state.final_prompt:
                    enhanced_styling = styling_prompt + "\n\n" + st.session_state.final_prompt if styling_prompt else st.session_state.final_prompt
                else:
                    enhanced_styling = styling_prompt
                
                # Use the selected model for image generation
                if ai_model == "Hybrid (OpenAI + Gemini)":
                    # Use the hybrid approach for best results
                    final_image, error, high_quality_fitted = hybrid_handler.fit_apparel_on_model(
                        st.session_state.apparel_image,
                        st.session_state.model_attributes,
                        "Natural standing",  # Default pose
                        model_image=st.session_state.model_image,
                        photoshoot_settings=st.session_state.photoshoot_settings,
                        styling_prompt=enhanced_styling,
                        apparel_color=st.session_state.apparel_color if "apparel_color" in st.session_state else None
                    )
                elif ai_model == "OpenAI DALL-E 3":
                    # Use OpenAI DALL-E 3 model
                    final_image, error, high_quality_fitted = dalle3_handler.fit_apparel_on_model(
                        st.session_state.apparel_image,
                        st.session_state.model_attributes,
                        "Natural standing",  # Default pose
                        model_image=st.session_state.model_image,
                        photoshoot_settings=st.session_state.photoshoot_settings,
                        styling_prompt=enhanced_styling,
                        apparel_color=st.session_state.apparel_color if "apparel_color" in st.session_state else None
                    )
                else:
                    # Fallback to Google's Imagen model
                    final_image, error, high_quality_fitted = imagen_handler.fit_apparel_on_model(
                        st.session_state.apparel_image,
                        st.session_state.model_attributes,
                        "Natural standing",  # Default pose
                        model_image=st.session_state.model_image,
                        photoshoot_settings=st.session_state.photoshoot_settings,
                        styling_prompt=enhanced_styling,
                        apparel_color=st.session_state.apparel_color if "apparel_color" in st.session_state else None
                    )
                
                if final_image:
                    st.session_state.final_image = final_image
                    # Store the high quality fitted image separately
                    st.session_state.high_quality_fitted = high_quality_fitted
                    # Set a flag to indicate photoshoot has been created
                    st.session_state.photoshoot_created = True
                    
                    # Show success message with the model used
                    if ai_model == "Hybrid (OpenAI + Gemini)":
                        st.success("Photoshoot created successfully with hybrid approach!")
                        st.info("The hybrid approach extracted detailed apparel features with OpenAI and created the try-on with Gemini for optimal results.")
                    else:
                        st.success(f"Photoshoot created successfully using {ai_model}!")
                    
                    st.image(final_image, caption=f"Final Photoshoot - {view_angle}", width="stretch")
                    
                    # Create columns for the download buttons
                    col1, col2 = st.columns(2)
                    
                    # Download option for the composite image
                    buf = io.BytesIO()
                    final_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    with col1:
                        st.download_button(
                            label="Download Full Composite",
                            data=byte_im,
                            file_name=f"virtual_photoshoot_composite_{int(time.time())}.png",
                            mime="image/png"
                        )
                    
                    # Download option for just the high-quality fitted image
                    if high_quality_fitted:
                        hq_buf = io.BytesIO()
                        high_quality_fitted.save(hq_buf, format="PNG", quality=100)
                        hq_byte_im = hq_buf.getvalue()
                        
                        with col2:
                            st.download_button(
                                label="Download HD Model Only",
                                data=hq_byte_im,
                                file_name=f"model_wearing_apparel_{int(time.time())}.png",
                                mime="image/png",
                                help="Download just the high-quality image of the model wearing the apparel"
                            )
                else:
                    st.error(f"Error creating photoshoot: {error}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 B+C Virtual Photoshoot App")

# Add information about the apparel requirements
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app allows you to fit your apparel onto one of our 10 predefined models. "
    "Select a model, customize with Bella+Canvas colors, and get realistic product-on-model renders."
)
