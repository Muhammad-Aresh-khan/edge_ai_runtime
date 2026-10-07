import sys
import os
import base64
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq import Groq
from src.config import DEFAULT_GROQ_API_KEY, DEFAULT_VLM_MODEL

client = Groq(api_key=DEFAULT_GROQ_API_KEY)
img_path = "data/samples/user_test_window_toddler.jpg"

with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

# Simulate YOLO candidate box (e.g. [543, 126, 757, 544] on a 1000x1000 normalized scale: [126, 543, 544, 757])
candidate_box = [126, 543, 544, 757]

prompt = f"""You are a child safety vision arbiter.
A lightweight detector proposed this candidate child bounding box in normalized [ymin, xmin, ymax, xmax] (0 to 1000 scale):
{candidate_box}

Task:
1. Is there a real human toddler/child in or near this candidate area? (valid: true/false)
2. If true, refine the exact bounding box of the child: [ymin, xmin, ymax, xmax].
3. If false (e.g. pillow/toy), set valid=false and if a child exists elsewhere in the image, provide their box; otherwise return null.

Return JSON ONLY:
{{
  "is_valid_child": true,
  "confidence": 0.95,
  "final_toddler_box_1000": [ymin, xmin, ymax, xmax],
  "reason": "..."
}}
"""

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
    max_tokens=150,
    response_format={"type": "json_object"}
)

print(res.choices[0].message.content)
