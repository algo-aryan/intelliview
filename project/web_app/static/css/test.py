import google.generativeai as genai

# ✅ Configure your API key here
genai.configure(api_key="AIzaSyA2Mi4IjnQf4TJ5FQyO3p21njnN7PRmDyg")  # <-- Replace with your actual key

try:
    models = genai.list_models()

    print("Available Gemini Models:\n")
    for model in models:
        print(f"Model Name: {model.name}")
        print(f"Supported Methods: {model.supported_generation_methods}")
        print("-" * 40)

except Exception as e:
    print("❌ Error fetching models:", e)