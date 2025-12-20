# Testing Gemini API with Postman

## Method 1: Check Docker Logs

Run this command to see which models initialized successfully:
```bash
docker compose logs api 2>&1 | grep -E "(Gemini|model)"
```

## Method 2: Direct API Test (Postman)

### Step 1: List Available Models
**Request:**
- Method: `GET`
- URL: `https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_API_KEY`
- Replace `YOUR_API_KEY` with your actual Google API key

**Expected Response:**
```json
{
  "models": [
    {
      "name": "models/gemini-pro",
      "displayName": "Gemini Pro",
      "supportedGenerationMethods": ["generateContent", "countTokens"]
    },
    ...
  ]
}
```

### Step 2: Test Content Generation
**Request:**
- Method: `POST`
- URL: `https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key=YOUR_API_KEY`
- Replace `{MODEL_NAME}` with a model from Step 1 (e.g., `gemini-pro`)
- Headers: `Content-Type: application/json`
- Body:
```json
{
  "contents": [
    {
      "parts": [
        {
          "text": "Hello! Say hi back in one word."
        }
      ]
    }
  ]
}
```

## Method 3: Using Python Script

I created `test_gemini.py` in your project root. Run:
```bash
source .venv/bin/activate
python test_gemini.py
```

This will:
1. Load your API key from `data/.env`
2. List all available models
3. Test with the first available model

## Common Model Names to Try:
- `models/gemini-pro`
- `models/gemini-1.5-pro`
- `models/gemini-1.5-flash`
- `models/gemini-1.5-flash-latest`

**Note:** The v1beta API sometimes has different model naming than the Python SDK.
