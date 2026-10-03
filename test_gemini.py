import os
from google import genai
from google.genai import types
from pydantic import BaseModel
import sys

print("Starting HTTP client test...", flush=True)

class DemoSchema(BaseModel):
    message: str

api_key = os.environ.get("GEMINI_API_KEY")
print(f"Key loaded: {api_key is not None}", flush=True)

try:
    # Use explicit http protocol or specify http options if it hangs on grpc
    client = genai.Client(api_key=api_key)
    print("Client initialized.", flush=True)

    print("Generating...", flush=True)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Say hello!",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DemoSchema,
            temperature=0.0,
        )
    )
    print(f"FINISH: {response.text}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr, flush=True)
