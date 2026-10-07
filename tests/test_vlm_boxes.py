import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import json
from groq import Groq
from src.config import DEFAULT_GROQ_API_KEY, DEFAULT_VLM_MODEL

client = Groq(api_key=DEFAULT_GROQ_API_KEY)
img_path = "data/samples/user_test_window_toddler.jpg"

with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

prompt = (
    "You are a household toddler safety visual grounding model.\n"
    "Identify any critical danger zone in this image (e.g. window, stairs, stove, balcony, sockets).\n"
    "For each danger zone, return its exact bounding box in normalized [ymin, xmin, ymax, xmax] coordinates from 0 to 1000.\n"
    "Output valid JSON ONLY with this schema:\n"
    '{"hazards": [{"label": "window", "box_1000": [ymin, xmin, ymax, xmax], "risk": "high"}]}'
)

res = client.chat.completions.create(
    model=DEFAULT_VLM_MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }],
    temperature=0.1,
    response_format={"type": "json_object"}
)

print(res.choices[0].message.content)
