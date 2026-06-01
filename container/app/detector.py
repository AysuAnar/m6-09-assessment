import os
import numpy as np
import onnxruntime as ort
from PIL import Image

class CatDetector:
    """
    Standardised CatDetector class matching the instructor's exact sketch.
    Loads models/best.onnx and performs high-precision inference using Pillow.
    """
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat", "dog", "cheetah")):
        # Load ONNX session
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")
            
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        
        # Get input and output shapes
        self.input_name = self.session.get_inputs()[0].name
        
        # Confirm output shape is (1, 300, 6) or matches e2e head
        output_shape = self.session.get_outputs()[0].shape
        print(f"Loaded ONNX model: {onnx_path}", file=os.sys.stderr)
        print(f"Input: {self.input_name}, Output Shape: {output_shape}", file=os.sys.stderr)

    def _letterbox(self, img, target_size=640):
        """
        Resizes and pads Pillow Image preserving aspect ratio.
        """
        orig_w, orig_h = img.size
        
        # Scale factor (r)
        scale = min(target_size / orig_w, target_size / orig_h)
        
        # New dimensions
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        
        # Support older/newer Pillow versions for BILINEAR resampling
        resampling = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        img_resized = img.resize((new_w, new_h), resampling)
        
        # Create padded background (114 gray)
        new_img = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        
        # Compute paddings
        pad_x = (target_size - new_w) / 2.0
        pad_y = (target_size - new_h) / 2.0
        
        # Paste resized image at padded coordinates
        new_img.paste(img_resized, (int(round(pad_x)), int(round(pad_y))))
        
        return new_img, scale, (pad_x, pad_y)

    def predict(self, image_path: str) -> list[dict]:
        """
        Performs inference on a single image and returns raw boxes scaled to original coordinates.
        """
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Letterbox to target size, keeping scale and paddings
        x_img, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        
        # Convert image to numpy float32, normalize, transpose to BCHW
        x = (np.array(x_img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Run ONNX inference
        out = self.session.run(None, {self.input_name: x})[0]  # YOLO26 e2e: (1, 300, 6)
        out = out[0]  # (300, 6) -> [x1, y1, x2, y2, score, class]

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue
                
            # Filter and keep ONLY Class 0 (cat) predictions
            if int(cls) != 0:
                continue
                
            # Undo letterbox padding and scaling
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            # Clip coordinates to original image bounds
            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))
            
            results.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
            
        return results
