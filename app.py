"""
SpectaAI / SafeSight Industrial PPE Vision Backend (FastAPI + YOLOv11 + WebSockets)

Loads your trained YOLOv11 PPE-detection model (best.pt) and exposes:
- WebSocket /ws/detect: Real-time live camera streaming & high-speed inference
- POST /detect: REST frame scanning endpoint (multipart & base64)
- GET /: Health, frontend static app, and model metadata

RUN LOCALLY:
  python app.py
  Runs at http://localhost:5000
"""

import os
import sys

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PPE-Backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app, load_yolo_model, model, model_classes

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" SpectaAI YOLOv11 PPE Neural Server Running on Port {port}")
    print(f" Web UI / API: http://localhost:{port}")
    print(f" WebSocket Stream: ws://localhost:{port}/ws/detect")
    print(f"=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
