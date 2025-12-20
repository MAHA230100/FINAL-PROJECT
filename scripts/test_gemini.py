#!/usr/bin/env python3
"""
Test script to list available Gemini models and test API connectivity.
Run this to see which models your API key has access to.

Usage:
    python test_gemini.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('data/.env')

try:
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        exit(1)
    
    print(f"✅ API Key loaded: {api_key[:10]}...")
    print("\n" + "="*60)
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    print("📋 Listing all available Gemini models:\n")
    
    # List all models
    models = genai.list_models()
    
    generative_models = []
    for m in models:
        print(f"Model: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Supported Methods: {m.supported_generation_methods}")
        print()
        
        if 'generateContent' in m.supported_generation_methods:
            generative_models.append(m.name)
    
    print("="*60)
    print(f"\n✅ Models supporting generateContent ({len(generative_models)}):")
    for model_name in generative_models:
        print(f"  - {model_name}")
    
    # Test with first available model
    if generative_models:
        print(f"\n🧪 Testing with model: {generative_models[0]}")
        model = genai.GenerativeModel(generative_models[0])
        response = model.generate_content("Hello! Say hi back in one word.")
        print(f"✅ Response: {response.text}")
    
except ImportError:
    print("❌ google-generativeai library not installed")
    print("Install with: pip install google-generativeai")
except Exception as e:
    print(f"❌ Error: {e}")
