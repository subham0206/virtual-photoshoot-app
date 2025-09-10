# B+C Virtual Photoshoot App

An AI-powered application that allows users to upload apparel and fit it onto customized virtual models. The app preserves the color, texture, and size of your apparel while placing it on a model with your chosen attributes.

## Features

- **Apparel Upload**: Upload your apparel image to be fitted onto a virtual model
- **Model Customization**: Customize models with specific attributes:
  - Gender (Male/Female)
  - Ethnicity
  - Height
  - Build/Physique
  - Hair color and style
  - Skin type
  - Bottom clothing style and color
  - Shoe style and color

- **Advanced Photoshoot Options**:
  - Various poses
  - Different view angles (front, back, side)
  - Background selection
  - Lighting adjustments
  - Quality settings

- **Multiple Variations**: Generate multiple photoshoot variations with different poses, backgrounds, or view angles

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/virtual-photoshoot-app.git
cd virtual-photoshoot-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage

1. Navigate to the "Upload Apparel" tab to upload your apparel image
2. Go to "Generate Model" to customize and create your AI model
3. Use the "Photoshoot" tab to fit your apparel on the model with custom settings
4. Download your final images

## Technologies Used

- Streamlit for the web interface
- Google's Generative AI models (Gemini 2.5 Flash Image)
- PIL/Pillow for image processing

## License

[MIT](https://opensource.org/licenses/MIT)