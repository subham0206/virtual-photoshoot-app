import os
import sys
import time
from PIL import Image
import io
from utils import ImagenHandler

# Check if we're running in a Streamlit environment and import if available
try:
    import streamlit as st
    USING_STREAMLIT = True
except ImportError:
    USING_STREAMLIT = False
    
# Set the API key
try:
    # Try to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
except:
    # Default to empty - will need to be provided
    api_key = ""

# If running in Streamlit environment, try to get key from Streamlit secrets
if USING_STREAMLIT:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass

# If no key found, prompt for it
if not api_key:
    print("No Google API key found in environment or Streamlit secrets.")
    api_key = input("Please enter your Google API key: ")

print("Initializing Imagen Handler...")

# Initialize with recommended GenAI models
imagen_handler = ImagenHandler(api_key, timeout=120, 
                              imagen_model="models/imagen-4.0-generate-001",
                              gemini_image_model="gemini-2.5-flash-image-preview")

# Define the 10 selected models with tweaks
MODELS_TO_CREATE = {
    "Male": [
        {
            "id": "m1",
            "name": "African American – Tall Athletic",
            "description": "Tall athletic build, fade haircut with curly top",
            "image_path": "models/male_african_american_tall.png",
            "prompt": "Create a hyper-realistic professional fashion model image of an African American male, 6'2\" tall with an athletic build, fade haircut with curly top. Model has defined muscles with broad shoulders. Wearing a plain white t-shirt and dark fitted jeans with minimalist sneakers. Neutral expression with confident posture. High-end studio photography with three-point lighting setup against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "m2",
            "name": "Spanish",
            "description": "Tan skin, slim build, brown buzz cut",
            "image_path": "models/male_spanish.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Spanish male with tan skin, slim build, and a brown buzz cut. Model has a more slender frame compared to athletic models. Wearing a plain white t-shirt and dark fitted jeans with minimalist sneakers. Neutral expression with relaxed posture. High-end studio photography with soft directional lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "m3",
            "name": "Japanese",
            "description": "Athletic build, dark brown buzz cut",
            "image_path": "models/male_japanese.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Japanese male with light skin, athletic build, and a dark brown buzz cut. Wearing a plain white t-shirt and dark fitted jeans with minimal white sneakers. Neutral expression with a poised stance. High-end studio photography with perfect lighting setup against a light grey backdrop. The model should have distinctly Asian features with a modern stylish appearance. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "m4",
            "name": "Middle Eastern",
            "description": "Tall slender build, buzz cut, light stubble",
            "image_path": "models/male_middle_eastern.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Middle Eastern male, 6'1\" tall with a slender athletic build, buzz cut (very short) black hair and light well-groomed stubble (not a beard). Wearing a plain white t-shirt and dark slim-fit jeans with leather casual shoes. Neutral expression with relaxed posture. High-end studio photography with soft directional lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "m5",
            "name": "Mixed Race",
            "description": "Tall athletic build, buzz cut, light stubble",
            "image_path": "models/male_mixed_race.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Mixed Race (Black/Caucasian) male, 6'2\" tall with an athletic build, dark brown buzz cut (no curls, just very short cropped hair) and light stubble. Wearing a plain white t-shirt and dark straight-leg jeans with white minimal sneakers. Neutral expression with a relaxed confident posture. High-end studio photography with balanced three-point lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        }
    ],
    "Female": [
        {
            "id": "f1",
            "name": "African American – Tall",
            "description": "Tall slender build, box braids",
            "image_path": "models/female_african_american_tall.png",
            "prompt": "Create a hyper-realistic professional fashion model image of an African American female, 5'11\" tall with a slender toned build, box braids in black hair (not natural curly hair). Wearing a plain white t-shirt and dark high-waisted jeans with minimal heeled boots. Neutral expression with elegant posture. High-end studio photography with soft beauty lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "f2",
            "name": "Middle Eastern",
            "description": "Tall slender build, long straight dark brown hair",
            "image_path": "models/female_middle_eastern.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Middle Eastern female, 5'10\" tall with a slender build, long straight dark brown hair (not windblown, but neatly styled). Wearing a plain white t-shirt and dark high-waisted jeans with minimal strappy sandals. Neutral expression with poised stance. High-end studio photography with soft diffused lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "f3",
            "name": "Mixed Race",
            "description": "Tall slender build, medium-length curly blonde hair",
            "image_path": "models/female_mixed_race.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Mixed Race (Asian/Caucasian) female, 5'11\" tall with a slender build, medium-length curly blonde hair (not honey-brown). Wearing a plain white t-shirt and dark fitted jeans with minimal ankle boots. Neutral expression with elegant stance. High-end studio photography with professional three-point lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "f4",
            "name": "South Asian",
            "description": "Tall slender build, shoulder-length wavy bob",
            "image_path": "models/female_south_asian.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a South Asian female, 5'10\" tall with a slender build, shoulder-length wavy bob haircut (not long straight hair reaching mid-back). Wearing a plain white t-shirt and dark fitted jeans with minimal heeled sandals. Neutral expression with graceful posture. High-end studio photography with perfect even lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        },
        {
            "id": "f5",
            "name": "Native American",
            "description": "Thin build, above sleeve length black hair",
            "image_path": "models/female_native_american.png",
            "prompt": "Create a hyper-realistic professional fashion model image of a Native American female with tan skin, thin build, and black hair cut to above sleeve length (shorter than the original long hair). Wearing a plain white t-shirt and dark jeans with casual shoes. Neutral expression with a natural standing pose. High-end studio photography with soft, even lighting against a light grey backdrop. Full body shot showing entire height with premium photographic quality. Ultra-detailed 4K resolution with professional fashion photography styling. The image must be extremely realistic, not cartoonish or AI-generated looking. Use imagen-4.0-generate-001 model for maximum photorealism."
        }
    ]
}

# Since we're only using the 10 selected models, we'll set the ADDITIONAL_MODELS to be empty
ADDITIONAL_MODELS = {
    "Male": [],
    "Female": []
}

def main():
    """Generate the model images"""
    
    # Determine whether to generate original models, additional models, or both
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created models directory.")
    
    # If running in Streamlit, use Streamlit UI
    if USING_STREAMLIT:
        st.title("Model Generation")
        
        # Add selection options in Streamlit
        mode = st.radio(
            "Select models to generate:",
            ["Original 10 Models", "New 10 Taller Models", "All 20 Models"],
            index=2
        )
        
        if mode == "Original 10 Models":
            mode = "original"
        elif mode == "New 10 Taller Models":
            mode = "additional"
        else:
            mode = "all"
        
        st.write(f"Generating {mode} models...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Determine which models to generate based on mode
    models_to_process = {}
    if mode in ["original", "all"]:
        for gender, models in MODELS_TO_CREATE.items():
            if gender not in models_to_process:
                models_to_process[gender] = []
            models_to_process[gender].extend(models)
    
    if mode in ["additional", "all"]:
        for gender, models in ADDITIONAL_MODELS.items():
            if gender not in models_to_process:
                models_to_process[gender] = []
            models_to_process[gender].extend(models)
    
    total_models = sum(len(models) for models in models_to_process.values())
    models_generated = 0
    
    for gender, models in models_to_process.items():
        print(f"\nGenerating {gender} models...")
        
        for model in models:
            model_id = model["id"]
            model_name = model["name"]
            image_path = model["image_path"]
            prompt = model["prompt"]
            
            # Skip if model already exists
            if os.path.exists(image_path):
                print(f"- Model {model_id} ({model_name}) already exists at {image_path}. Skipping.")
                models_generated += 1
                
                # Update progress in Streamlit
                if USING_STREAMLIT:
                    progress_bar.progress(models_generated / total_models)
                    status_text.text(f"Skipped {model_name} (already exists)")
                
                continue
            
            print(f"- Generating model {model_id} ({model_name})...")
            if USING_STREAMLIT:
                status_text.text(f"Generating {model_name}...")
            
            # Generate the model image using the prompt
            try:
                model_image = imagen_handler.generate_image_with_api(prompt)
                
                # Save the image
                if model_image:
                    model_image.save(image_path)
                    print(f"  Successfully generated and saved to {image_path}")
                    
                    # Show the image in Streamlit
                    if USING_STREAMLIT:
                        st.image(model_image, caption=f"{model_name}", width=300)
                else:
                    print(f"  Failed to generate model {model_id}")
                    if USING_STREAMLIT:
                        st.error(f"Failed to generate model {model_name}")
            
            except Exception as e:
                print(f"  Error generating model {model_id}: {str(e)}")
                if USING_STREAMLIT:
                    st.error(f"Error generating {model_name}: {str(e)}")
            
            # Increment counter and update progress
            models_generated += 1
            if USING_STREAMLIT:
                progress_bar.progress(models_generated / total_models)
            
            # Brief pause to avoid rate limiting
            time.sleep(2)
    
    print(f"\nGeneration complete! {models_generated}/{total_models} models processed.")
    if USING_STREAMLIT:
        st.success(f"Generation complete! {models_generated}/{total_models} models processed.")
        st.balloons()

if __name__ == "__main__":
    main()