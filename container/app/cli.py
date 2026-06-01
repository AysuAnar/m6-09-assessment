import os
import sys

# Add parent directory of 'app' to sys.path to support running as a standalone script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import csv
import json
import argparse
from app.detector import CatDetector

def run_info():
    """
    Prints STUDENT.json to stdout. Used by the leaderboard runner to register the entry.
    Expected output: the contents of /app/STUDENT.json, valid JSON, on stdout, exit code 0.
    """
    # Find STUDENT.json in same folder as cli.py or fallback candidates
    candidates = [
        os.path.join(os.path.dirname(__file__), "STUDENT.json"),
        os.path.join(os.path.dirname(__file__), "..", "STUDENT.json"),
        "STUDENT.json",
        "/app/STUDENT.json"
    ]
    
    student_json_path = None
    for path in candidates:
        if os.path.exists(path):
            student_json_path = path
            break
            
    if not student_json_path:
        print("Error: STUDENT.json not found.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(student_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Print to stdout in valid JSON format
            print(json.dumps(data, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"Error reading STUDENT.json: {e}", file=sys.stderr)
        sys.exit(1)

def run_predict(input_dir, output_file, conf_thres, model_path_arg):
    """
    Runs inference on a folder of images and writes a CSV of bounding-box predictions.
    """
    # 1. Locate ONNX model
    model_candidates = [
        model_path_arg,
        os.path.join(os.path.dirname(__file__), "..", "models", "best.onnx"),
        os.path.join("models", "best.onnx"),
        "best.onnx",
        "/models/best.onnx"
    ]
    
    model_path = None
    for path in model_candidates:
        if path and os.path.exists(path):
            model_path = path
            break
            
    if not model_path:
        print("Error: Could not locate ONNX model (best.onnx). Ensure it exists in models/.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Initialize detector
    try:
        detector = CatDetector(model_path, conf=conf_thres)
    except Exception as e:
        print(f"Error initializing detector: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Recursively find all images in the input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    image_files = []
    extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                # Compute relative path using forward slashes as requested by schema
                rel_path = os.path.relpath(full_path, start=input_dir).replace("\\", "/")
                image_files.append((full_path, rel_path))
                
    # Sort by relative path for deterministic output
    image_files.sort(key=lambda x: x[1])
    
    print(f"Found {len(image_files)} images to process recursively.", file=sys.stderr)
    
    # 4. Prepare output CSV predictions list
    # Header: image_path,xmin,ymin,xmax,ymax,confidence,class
    csv_rows = []
    
    for idx, (full_path, rel_path) in enumerate(image_files):
        try:
            detections = detector.predict(full_path)
            if detections:
                for det in detections:
                    xmin = det["xmin"]
                    ymin = det["ymin"]
                    xmax = det["xmax"]
                    ymax = det["ymax"]
                    confidence = det["confidence"]
                    cls_name = det["class"]
                    
                    csv_rows.append([
                        rel_path,
                        round(xmin, 1),
                        round(ymin, 1),
                        round(xmax, 1),
                        round(ymax, 1),
                        round(confidence, 2),
                        cls_name
                    ])
                print(f"[{idx+1}/{len(image_files)}] Processed {rel_path} - {len(detections)} detections", file=sys.stderr)
            else:
                # No detections: write single row with image_path filled and others empty
                csv_rows.append([rel_path, "", "", "", "", "", ""])
                print(f"[{idx+1}/{len(image_files)}] Processed {rel_path} - No detections", file=sys.stderr)
        except Exception as e:
            # Handle image loading or processing failure gracefully by outputting an empty detection row
            print(f"[{idx+1}/{len(image_files)}] Error processing {rel_path}: {e}", file=sys.stderr)
            csv_rows.append([rel_path, "", "", "", "", "", ""])
            
    # 5. Write predictions to output CSV
    try:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write standardized header
            writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"])
            writer.writerows(csv_rows)
            
        print(f"Successfully wrote predictions.csv to {output_file}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error writing output CSV file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="YOLO26m Standardised CLI for Leaderboard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # info subcommand
    subparsers.add_parser("info", help="Prints STUDENT.json to stdout")
    
    # predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Runs predictions and writes predictions.csv")
    predict_parser.add_argument("-i", "--input", default="/data/input", help="Path to input directory of images")
    predict_parser.add_argument("-o", "--output", default="/data/output/predictions.csv", help="Path to output predictions.csv")
    predict_parser.add_argument("-c", "--conf", type=float, default=0.25, help="Confidence threshold")
    predict_parser.add_argument("-m", "--model", default=None, help="Optional path to model file")
    
    # In case the command is passed directly (fallback handling)
    # Parse args
    args = parser.parse_args()
    
    if args.command == "info":
        run_info()
    elif args.command == "predict":
        run_predict(args.input, args.output, args.conf, args.model)

if __name__ == "__main__":
    main()
