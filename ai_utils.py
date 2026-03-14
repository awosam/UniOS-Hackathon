import os
import warnings
from dotenv import load_dotenv
import google.generativeai as genai

# Silence Python 3.9 EOL warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# Configure the Brain
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def ask_uni_os(prompt, use_pro=False):
    """
    The gateway to Uni-OS logic.
    Use PRO only for policy analysis or complex major-switching logic.
    """
    model_name = "gemini-1.5-pro" if use_pro else "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Brain Connection Error: {str(e)}"

if __name__ == "__main__":
    print(f"--- Brain Online ---")
    print(ask_uni_os("Hello! This is Uni-OS. Are we ready to build?"))