import streamlit as st
import os
import io
from PIL import Image
import time
from utils import ImagenHandler, extract_apparel_features
import socket
import sys

# Set the API key
try:
    # Try to get API key from Streamlit secrets (for deployment)
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    # Fallback to hardcoded key for local development
    api_key = ""

# Initialize with recommended GenAI models: "imagen-4.0-generate-001" and "gemini-2.5-flash-image-preview"
imagen_handler = ImagenHandler(api_key, timeout=90, 
                              imagen_model="models/imagen-4.0-generate-001",
                              gemini_image_model="gemini-2.5-flash-image-preview")

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

# Create sidebar for app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Apparel", "Generate Model", "Photoshoot"])

# Global session state to store uploaded image and generated model
if "apparel_image" not in st.session_state:
    st.session_state.apparel_image = None
if "model_image" not in st.session_state:
    st.session_state.model_image = None
if "final_image" not in st.session_state:
    st.session_state.final_image = None
if "model_attributes" not in st.session_state:
    st.session_state.model_attributes = {}
if "photoshoot_variations" not in st.session_state:
    st.session_state.photoshoot_variations = []
if "network_retry_count" not in st.session_state:
    st.session_state.network_retry_count = 0

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
        
        # Add color change option
        with st.expander("Change Apparel Color", expanded=True):
            st.info("Use this option if you want to change the color of your apparel.")
            color_change = st.checkbox("I want to change the apparel color")
            
            if color_change:
                # Color selection options
                color_method = st.radio("Color Selection Method", ["Color Picker", "Hex Code Input"])
                
                if color_method == "Color Picker":
                    selected_color = st.color_picker("Select New Color", "#0000FF")
                else:
                    selected_color = st.text_input("Enter Hex Color Code", "#0000FF")
                    # Validate hex code
                    if selected_color and not selected_color.startswith('#'):
                        selected_color = '#' + selected_color
                
                # Store the selected color in session state
                if "apparel_color" not in st.session_state:
                    st.session_state.apparel_color = None
                
                # Apply color button
                if st.button("Apply Color"):
                    with st.spinner("Applying color..."):
                        try:
                            # Store the color for use in the generation step
                            st.session_state.apparel_color = selected_color
                            st.success(f"Color {selected_color} will be applied during photoshoot generation.")
                        except Exception as e:
                            st.error(f"Error applying color: {str(e)}")
            else:
                # Reset the color if checkbox is unchecked
                st.session_state.apparel_color = None
        
        # Use 'content' instead of 'auto' for width
        st.image(image, caption="Uploaded Apparel Image", width="content")
        st.success("Apparel image uploaded successfully!")

# Generate Model page
elif page == "Generate Model":
    st.header("Generate AI Model")
    
    if st.session_state.apparel_image is None:
        st.warning("Please upload an apparel image first.")
    else:
        # Display the uploaded apparel
        st.subheader("Uploaded Apparel")
        st.image(st.session_state.apparel_image, width=300)
        
        # Model customization options
        st.subheader("Customize Your Model")
        
        # Gender selection first to determine which options to show
        gender = st.radio("Select Model Gender", ["Male", "Female"])
        
        # Add facial expression selection
        facial_expression = st.selectbox(
            "Facial Expression",
            options=["Neutral", "Smiling", "Serious", "Confident", "Relaxed"]
        )
        
        # Define styling options based on gender
        if gender == "Male":
            col1, col2 = st.columns(2)
            
            with col1:
                ethnicity = st.selectbox(
                    "Ethnicity",
                    options=["African American", "Caucasian", "Spanish", "Korean", "Japanese"]
                )
                
                height = st.selectbox(
                    "Height",
                    options=["6'0\"", "6'1\"", "6'2\""]
                )
                
                build = st.selectbox(
                    "Build",
                    options=["Athletic", "Slender"]
                )
            
            with col2:
                hair_color = st.selectbox(
                    "Hair Color",
                    options=["Brown", "Black", "Bleach Blonde"]
                )
                
                hair_style = st.selectbox(
                    "Hair Style",
                    options=["Buzz Cut", "Wavy", "Curly", "Tapered Haircut"]
                )
                
                skin = st.selectbox(
                    "Skin Type",
                    options=["Smooth", "Smooth with Moles"]
                )
            
            # Additional styling options in expandable sections
            with st.expander("Bottoms Style"):
                bottom_color = st.selectbox(
                    "Bottom Color",
                    options=["Black", "Dark Denim", "Light Denim", "Faded Denim", "Navy", "White", "Off-White"]
                )
                
                bottom_style = st.selectbox(
                    "Bottom Style",
                    options=["Straight Leg Denim", "Baggy Denim", "Utility Denim", "BC Sweatpants"]
                )
            
            with st.expander("Shoes Style"):
                shoe_color = st.selectbox(
                    "Shoe Color",
                    options=["Black", "White", "Metallic Silver", "Grey", "Navy", "Multicolored"]
                )
                
                shoe_style = st.selectbox(
                    "Shoe Style",
                    options=["Loafer", "New Balance 1906R", "Converse Chuck Taylor"]
                )
        
        else:  # Female options
            col1, col2 = st.columns(2)
            
            with col1:
                ethnicity = st.selectbox(
                    "Ethnicity",
                    options=["African American", "Caucasian", "Spanish", "Korean", "Japanese"]
                )
                
                height = st.selectbox(
                    "Height",
                    options=["5'9\"", "5'10\"", "5'11\""]
                )
                
                build = st.selectbox(
                    "Build",
                    options=["Thin", "Toned", "Small Bust"]
                )
            
            with col2:
                hair_color = st.selectbox(
                    "Hair Color",
                    options=["Blonde", "Brown", "Black", "Brown with Blonde Highlights"]
                )
                
                hair_style = st.selectbox(
                    "Hair Style",
                    options=[
                        "Down with Soft Wave & Center Part", 
                        "Down with Center Part & Straight", 
                        "Loose Low Bun with Center Part", 
                        "Slicked Back High Ponytail with Braid",
                        "Slicked Back Low Ponytail"
                    ]
                )
                
                skin = st.selectbox(
                    "Skin Type",
                    options=["Smooth", "Smooth with Moles"]
                )
            
            # Additional styling options in expandable sections
            with st.expander("Bottoms Style"):
                bottom_color = st.selectbox(
                    "Bottom Color",
                    options=["Black", "Dark Denim", "Light Denim", "Faded Denim", "Navy", "White", "Off-White"]
                )
                
                bottom_style = st.selectbox(
                    "Bottom Style",
                    options=["Straight Leg Denim", "Baggy Denim", "Utility Denim", "BC Sweatpants"]
                )
            
            with st.expander("Shoes Style"):
                shoe_color = st.selectbox(
                    "Shoe Color",
                    options=["White", "Off-White", "Black", "Metallic Silver", "Grey", "Navy", "Multicolored"]
                )
                
                shoe_style = st.selectbox(
                    "Shoe Style",
                    options=["Converse Chuck Taylor High Top", "Loafers", "Adidas Gazelle", "New Balance 237"]
                )
        
        # Store model attributes with all the detailed styling options
        model_attributes = {
            "gender": gender,
            "ethnicity": ethnicity,
            "height": height,
            "build": build,
            "hair_color": hair_color,
            "hair_style": hair_style,
            "skin": skin,
            "bottom_color": bottom_color,
            "bottom_style": bottom_style,
            "shoe_color": shoe_color,
            "shoe_style": shoe_style,
            "facial_expression": facial_expression
        }
        
        # Generate model button
        if st.button("Generate Model"):
            with st.spinner("Generating model... This may take a moment."):
                # Create a detailed description for the model generation
                detailed_description = f"{gender} model, {ethnicity}, {height} tall, {build.lower()} build, "
                detailed_description += f"{hair_color.lower()} {hair_style.lower()} hair, {skin.lower()} skin, "
                detailed_description += f"wearing {bottom_color.lower()} {bottom_style.lower()} and {shoe_color.lower()} {shoe_style.lower()} shoes."
                
                # Use our ImagenHandler to generate the model image with the detailed attributes
                model_image, error = imagen_handler.generate_model_image(
                    gender=gender,
                    physique=build,
                    ethnicity=ethnicity,
                    height=height,
                    hair_color=hair_color,
                    hair_style=hair_style,
                    skin=skin,
                    clothing_type=f"{bottom_color} {bottom_style}",
                    shoe_style=f"{shoe_color} {shoe_style}",
                    facial_expression=facial_expression
                )
                
                if model_image:
                    st.session_state.model_image = model_image
                    st.session_state.model_attributes = model_attributes
                    st.success("Model generated successfully!")
                    # Use 'content' instead of 'auto' for width
                    st.image(model_image, caption=f"Generated {gender} Model: {detailed_description}", width="content")
                    st.markdown("Now go to the 'Photoshoot' tab to fit your apparel on this model.")
                else:
                    st.error(f"Error generating model: {error}")

elif page == "Photoshoot":
    st.header("Virtual Photoshoot")
    
    if st.session_state.apparel_image is None:
        st.warning("Please upload an apparel image first.")
    elif st.session_state.model_image is None:
        st.warning("Please generate a model first.")
    else:
        # Display the uploaded apparel and generated model
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Apparel")
            st.image(st.session_state.apparel_image, width=300)
        
        with col2:
            st.subheader("Your Model")
            st.image(st.session_state.model_image, width=300)
        
        # Create tabs for different photoshoot configuration options
        pose_tab, view_tab, background_tab, lighting_tab = st.tabs(["Pose", "View Angle", "Background", "Lighting"])
        
        # Pose selection tab
        with pose_tab:
            st.subheader("Select Pose")
            pose_options = [
                "Natural standing", "Casual walking", "Seated", "Arms crossed",
                "Hand on hip", "Hands in pockets", "Runway walk", "Looking over shoulder",
                "Profile view", "Jumping", "Action pose"
            ]
            
            pose = st.selectbox("Choose a pose", options=pose_options)
        
        # View angle tab
        with view_tab:
            st.subheader("Select View Angle")
            view_options = [
                "Front view", "Back view", "Side view (left)", "Side view (right)", 
                "Three-quarter front", "Three-quarter back", "Low angle", "High angle"
            ]
            
            view_angle = st.selectbox("Choose a view angle", options=view_options)
        
        # Background tab
        with background_tab:
            st.subheader("Select Background")
            background_options = [
                "Studio white", "Studio light gray", "Studio dark gray", "Studio black",
                "Minimalist interior", "Urban street", "Nature outdoor", "Beach", 
                "Gradient", "Solid color", "Transparent (for e-commerce)"
            ]
            
            background = st.selectbox("Choose a background", options=background_options)
            
            # Show additional options based on background selection
            if background == "Gradient":
                gradient_direction = st.selectbox(
                    "Gradient Direction", 
                    options=["Top to Bottom", "Left to Right", "Diagonal"]
                )
                gradient_colors = st.selectbox(
                    "Gradient Colors",
                    options=["Blue to Purple", "Orange to Pink", "Green to Blue", "Gray to White", "Custom"]
                )
                if gradient_colors == "Custom":
                    col1, col2 = st.columns(2)
                    with col1:
                        color1 = st.color_picker("First Color", "#ffffff")
                    with col2:
                        color2 = st.color_picker("Second Color", "#e0e0e0")
            
            elif background == "Solid color":
                background_color = st.color_picker("Choose background color", "#f0f0f0")
        
        # Lighting tab
        with lighting_tab:
            st.subheader("Lighting Settings")
            lighting_style = st.selectbox(
                "Lighting Style",
                options=[
                    "Standard studio", "Soft diffused", "Dramatic", "Bright high-key", 
                    "Dark low-key", "Natural sunlight", "Golden hour", "Blue hour"
                ]
            )
            
            lighting_direction = st.selectbox(
                "Main Light Direction",
                options=["Front", "Side", "Back (rim lighting)", "Top-down", "Bottom-up", "Three-point lighting"]
            )
            
            # Advanced lighting controls with sliders
            with st.expander("Advanced Lighting Controls"):
                light_intensity = st.slider("Light Intensity", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
                contrast = st.slider("Contrast", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                shadows = st.slider("Shadow Intensity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            generate_multiple = st.checkbox("Generate multiple variations")
            
            if generate_multiple:
                num_variations = st.slider("Number of variations", min_value=1, max_value=5, value=3)
                variation_type = st.radio("Variation type", ["Pose variations", "Background variations", "View angle variations"])
                
                if variation_type == "Pose variations":
                    selected_poses = st.multiselect(
                        "Select poses for variations",
                        options=pose_options,
                        default=[pose_options[0], pose_options[1], pose_options[2]]
                    )
                elif variation_type == "Background variations":
                    selected_backgrounds = st.multiselect(
                        "Select backgrounds for variations",
                        options=background_options,
                        default=[background_options[0], background_options[1], background_options[2]]
                    )
                else:  # View angle variations
                    selected_views = st.multiselect(
                        "Select view angles for variations",
                        options=view_options,
                        default=[view_options[0], view_options[1], view_options[2]]
                    )
            
            # Add quality settings
            st.subheader("Quality Settings")
            quality_options = st.radio(
                "Image Quality", 
                ["Standard", "High", "Ultra HD (slower generation)"],
                horizontal=True
            )
        
        # Store photoshoot settings in session state
        if "photoshoot_settings" not in st.session_state:
            st.session_state.photoshoot_settings = {}
        
        # Update photoshoot settings based on user selections
        st.session_state.photoshoot_settings = {
            "pose": pose,
            "view_angle": view_angle,
            "background": background,
            "lighting_style": lighting_style,
            "lighting_direction": lighting_direction
        }
        
        # Add background specific settings
        if background == "Gradient" and "gradient_colors" in locals():
            st.session_state.photoshoot_settings["gradient_direction"] = gradient_direction
            st.session_state.photoshoot_settings["gradient_colors"] = gradient_colors
            if gradient_colors == "Custom" and "color1" in locals():
                st.session_state.photoshoot_settings["gradient_color1"] = color1
                st.session_state.photoshoot_settings["gradient_color2"] = color2
        elif background == "Solid color" and "background_color" in locals():
            st.session_state.photoshoot_settings["background_color"] = background_color
        
        # Add advanced lighting settings if expanded
        if "light_intensity" in locals():
            st.session_state.photoshoot_settings["light_intensity"] = light_intensity
            st.session_state.photoshoot_settings["contrast"] = contrast
            st.session_state.photoshoot_settings["shadows"] = shadows
            
        # Add quality settings
        st.session_state.photoshoot_settings["quality"] = quality_options
        
        # Fit apparel button
        if st.button("Fit Apparel & Create Photoshoot"):
            if generate_multiple and (('selected_poses' in locals() and len(selected_poses) > 0) or 
                               ('selected_backgrounds' in locals() and len(selected_backgrounds) > 0) or
                               ('selected_views' in locals() and len(selected_views) > 0)):
                with st.spinner("Creating multiple photoshoot variations... This may take a moment."):
                    # Determine which type of variation we're generating
                    if variation_type == "Pose variations" and 'selected_poses' in locals():
                        variations = selected_poses
                        var_type = "pose"
                    elif variation_type == "Background variations" and 'selected_backgrounds' in locals():
                        variations = selected_backgrounds
                        var_type = "background"
                    elif variation_type == "View angle variations" and 'selected_views' in locals():
                        variations = selected_views
                        var_type = "view"
                    else:
                        st.error("Please select at least one variation option")
                        # Instead of 'return', we'll use a boolean variable to control the flow
                        should_continue = False
                    
                    # Generate multiple variations with the current photoshoot settings
                    images, error = imagen_handler.generate_photoshoot_variations(
                        st.session_state.apparel_image,
                        st.session_state.model_attributes,
                        variations,
                        count=num_variations,
                        model_image=st.session_state.model_image,
                        variation_type=var_type,
                        photoshoot_settings=st.session_state.photoshoot_settings
                    )
                    
                    if images and len(images) > 0:
                        st.session_state.photoshoot_variations = images
                        st.success(f"Successfully generated {len(images)} photoshoot variations!")
                        
                        # Display all variations
                        st.subheader("Your Photoshoot Variations")
                        cols = st.columns(min(len(images), 3))
                        
                        for i, image in enumerate(images):
                            with cols[i % len(cols)]:
                                # Create caption based on variation type
                                if variation_type == "Pose variations":
                                    caption = f"Pose: {variations[i]}"
                                elif variation_type == "Background variations":
                                    caption = f"Background: {variations[i]}"
                                else:  # View angle variations
                                    caption = f"View: {variations[i]}"
                                    
                                st.image(image, caption=caption, width="stretch")
                                
                                # Download button for each variation
                                buf = io.BytesIO()
                                image.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label=f"Download Variation {i+1}",
                                    data=byte_im,
                                    file_name=f"virtual_photoshoot_var{i+1}_{int(time.time())}.png",
                                    mime="image/png"
                                )
                    else:
                        st.error(f"Error creating photoshoot variations: {error}")
            else:
                with st.spinner("Creating your photoshoot... This may take a moment."):
                    # Fit apparel on model using our ImagenHandler - passing the model image and photoshoot settings
                    final_image, error, high_quality_fitted = imagen_handler.fit_apparel_on_model(
                        st.session_state.apparel_image,
                        st.session_state.model_attributes,
                        pose,
                        model_image=st.session_state.model_image,  # Pass the model image
                        photoshoot_settings=st.session_state.photoshoot_settings  # Pass photoshoot settings
                    )
                    
                    if final_image:
                        st.session_state.final_image = final_image
                        # Store the high quality fitted image separately
                        st.session_state.high_quality_fitted = high_quality_fitted
                        
                        st.success("Photoshoot created successfully!")
                        st.image(final_image, caption=f"Final Photoshoot - {pose} Pose", width="stretch")
                        
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
    "This app preserves the color, texture, and size of your apparel "
    "when fitting it onto AI-generated models. Perfect for virtual try-ons "
    "and e-commerce applications."
)
