import os
import google.generativeai as genai

# Set the API key
api_key = ""
genai.configure(api_key=api_key)

# List available models
print("Checking available models...")
try:
    model_list = genai.list_models()
    print("\nAvailable models:")
    for model in model_list:
        print(f"- {model.name} (supported generation methods: {', '.join(model.supported_generation_methods)})")
except Exception as e:
    print(f"Error checking models: {e}")
