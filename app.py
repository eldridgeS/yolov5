from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import onnxruntime
import numpy as np
from pathlib import Path
import json
import base64
import time
from models.common import DetectMultiBackend 
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device 

#Initialize Flask app
app = Flask(__name__)

MODEL_PATH = Path('runs/train/waste_detector11/weights/best.onnx')

#Settings
CONF_THRES = 0.25  # Confidence threshold
IOU_THRES = 0.45   # IoU threshold for NMS
IMG_SIZE = 640     # Model input image size (e.g., 640x640)

# Load model
try:
    # Use CPUExecutionProvider as we are running on CPU
    session = onnxruntime.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    
    # Get input and output names from the ONNX model
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()] # Handle multiple outputs if needed
    
    # Get class names from your data.yaml. Assuming data.yaml is next to app.py
    # Or, you can hardcode them here if they are fixed.
    # We will assume you manually list them for simplicity, as data.yaml pathing can be tricky here.
    CLASS_NAMES = ['aerosol_can', 'b', 'cardboard_box', 'clothing',
     'coffee_grounds', 'cosmetic_container', 'detergent_bottle',
      'egg_shells', 'food_can', 'food_jar', 'food_waste', 'glass_bottle',
       'magazine', 'newspaper', 'objects', 'packaging', 'paper', 'paper_cup',
        'plastic_bag', 'plastic_bottle', 'plastic_cup', 'plastic_cup_lid',
         'plastic_cutlery', 'plastic_food_container', 'shoes', 'soda_bottle',
          'soda_can', 'straws', 'styrofoam_cup', 'styrofoam_food_container', 'tea_bag', 'trash_bag']
    
    print(f"ONNX model loaded from {MODEL_PATH}")
    print(f"Model input name: {input_name}")
    print(f"Model output names: {output_names}")

except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None
    CLASS_NAMES = [] # Ensure it's empty if model fails to load

#Object dictionary and info
OBJECT_INFO = {
    # Metals
    "aerosol_can": "RECYCLE: Metal aerosol cans. Ensure empty. Check local rules if punctured.",
    "food_can": "RECYCLE: Metal food cans (steel/tin). Rinse thoroughly.",
    "soda_can": "RECYCLE: Aluminum soda cans. Rinse if sticky.",

    # Paper & Cardboard
    "cardboard_box": "RECYCLE: Cardboard boxes. Flatten. Keep dry.",
    "magazine": "RECYCLE: Magazines. Dispose of with mixed paper.",
    "newspaper": "RECYCLE: Newspapers. Dispose of with mixed paper.",
    "paper": "RECYCLE: Clean, dry mixed paper (e.g., mail, office paper).",

    # Plastics (Strictly #1 & #2 Bottles/Jugs ONLY for curbside in Atlanta)
    "detergent_bottle": "RECYCLE: Plastic detergent bottles (#2 HDPE). Rinse well.",
    "plastic_bottle": "RECYCLE: Plastic bottles (#1 PET, #2 HDPE). Rinse. Caps can often be left on or discarded based on size.",
    "soda_bottle": "RECYCLE: Plastic soda bottles (#1 PET). Rinse if sticky.",
    
    # Glass
    "food_jar": "RECYCLE: Glass food jars. Rinse well. Remove lids.",
    "glass_bottle": "RECYCLE: Glass bottles. Rinse well. Remove caps.",

    # Items Generally NOT Accepted in Atlanta Curbside Recycling (TRASH)
    "clothing": "TRASH: Clothing (unless donating or using textile recycling programs).",
    "coffee_grounds": "TRASH: Coffee grounds (compost if possible).",
    "cosmetic_container": "TRASH: Cosmetic containers (often mixed materials or non-#1/#2 plastics).",
    "egg_shells": "TRASH: Egg shells (compost if possible).",
    "food_waste": "TRASH: Food waste (compost if possible).",
    "objects": "TRASH: Generic items. Disposal depends on specific material and local rules.",
    "packaging": "TRASH: Generic packaging. Disposal depends on specific material and local rules (often TRASH if not specified recyclable type).", 
    "paper_cup": "TRASH: Paper cups (often have plastic/wax lining, not accepted).",
    "plastic_bag": "TRASH: Plastic bags (require special store drop-off, not curbside).",
    "plastic_cup": "TRASH: Plastic cups (typically not #1/#2 bottle/jug shape, not accepted).",
    "plastic_cup_lid": "TRASH: Plastic cup lids (often too small/thin, not accepted).",
    "plastic_cutlery": "TRASH: Plastic cutlery (too small, not accepted).",
    "plastic_food_container": "TRASH: Plastic food containers (e.g., clam shells, yogurt tubs, often not #1/#2 bottle/jug, generally not accepted).",
    "shoes": "TRASH: Shoes (unless donating or using shoe recycling programs).",
    "straws": "TRASH: Straws (too small, not accepted).",
    "styrofoam_cup": "TRASH: Styrofoam cups (not accepted).",
    "styrofoam_food_container": "TRASH: Styrofoam food containers (not accepted).",
    "tea_bag": "TRASH: Tea bags (compost if plastic-free, otherwise general waste).",
    "trash_bag": "TRASH: Trash bags (intended for general waste).",

    # Defaults
    "default": "DISPOSAL: Check local guidelines for this item.",
    "no_detection": "No objects were detected in the image."
}

#Setup Camera
camera = cv2.VideoCapture(0) # Default USB camera

if not camera.isOpened():
    print("Error: Could not open camera. Ensure it's connected and drivers are okay.")

#Functions

def get_object_info(class_name):
    """Retrieves pre-written information for a given class name."""
    return OBJECT_INFO.get(class_name.lower(), OBJECT_INFO["default"])

def preprocess_image_for_onnx(image_np, img_size):
    """
    Preprocesses a NumPy image for ONNX model input.
    Resizes, normalizes, transposes, and adds batch dimension.
    """
    original_height, original_width = image_np.shape[:2]
    
    # Resize image to model input size (e.g., 640x640)
    resized_image = cv2.resize(image_np, (img_size, img_size))
    
    # Normalize (0-255 to 0-1)
    normalized_image = resized_image / 255.0
    
    # Transpose from HWC to CHW (Height, Width, Channels to Channels, Height, Width)
    transposed_image = np.transpose(normalized_image, (2, 0, 1))
    
    # Add batch dimension (CHW to 1CHW)
    input_tensor = np.expand_dims(transposed_image, axis=0)
    
    # Convert to float32
    input_tensor = input_tensor.astype(np.float32)
    
    return input_tensor, (original_height, original_width)

def process_detections_from_onnx_output(onnx_output, original_img_shape, img_size):
    """
    Processes raw ONNX model output to get filtered detections.
    Applies NMS and scales bounding boxes to original image dimensions.
    """
    # ONNX output is usually a list of arrays, take the first element (the predictions)
    predictions_np = onnx_output[0]
    
    # Convert numpy array to torch tensor for NMS
    predictions_tensor = torch.tensor(predictions_np)

    # Apply Non-Maximum Suppression (NMS)
    # This function expects [batch_size, num_detections, 5 + num_classes] format
    # and returns a list of tensors, one per image in batch, with [x1, y1, x2, y2, conf, cls]
    pred = non_max_suppression(predictions_tensor, CONF_THRES, IOU_THRES, classes=None, agnostic=False, max_det=1000)
    
    # Get detections for the first (and only) image in our batch
    detections = pred[0] # detections is a tensor: [x1, y1, x2, y2, conf, cls]
    
    # Scale bounding boxes from model's input size (img_size) back to original image size
    # This is critical for drawing correct boxes on the original frame
    scaled_detections = []
    if len(detections):
        # The scale_boxes function from yolov5 expects detections in xyxy format
        # and scales them in-place or returns a new tensor.
        # We need the original image shape, and the shape the model was trained on (img_size, img_size)
        # Note: The 'original_img_shape' here means the shape of the image fed to `preprocess_image_for_onnx`
        # and 'img_size' is the model's expected input resolution.
        scaled_detections_tensor = scale_boxes(
            (img_size, img_size), # shape_of_model_input (h,w)
            detections[:, :4], # boxes in xyxy format
            original_img_shape # shape_of_original_image (h,w)
        )
        # Combine scaled boxes with confidence and class
        scaled_detections = torch.cat((scaled_detections_tensor, detections[:, 4:]), dim=1)

    return scaled_detections.cpu().numpy() # Convert to numpy for easier processing and drawing

def draw_detections(image_np, detections):
    """
    Draws bounding boxes and labels on the image for display.
    Expects detections in numpy array: [x1, y1, x2, y2, conf, cls_id]
    """
    processed_image = image_np.copy()
    detected_class_names = []

    if detections is None or len(detections) == 0:
        return processed_image, "no_detection"

    for *xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        confidence = float(conf)
        class_id = int(cls_id)
        
        if class_id < len(CLASS_NAMES): # Ensure class_id is within bounds
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = "unknown" # Fallback for out-of-bounds class_id

        detected_class_names.append(class_name)

        # Draw rectangle
        color = (0, 255, 0) # Green BGR
        thickness = 2
        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, thickness)

        # Draw label background and text
        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 5
        cv2.rectangle(processed_image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), color, -1)
        cv2.putText(processed_image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Compile and return information for all detected classes
    if detected_class_names:
        all_info = []
        for class_name in set(detected_class_names): # Use set to get unique class names
            all_info.append(f"{class_name}: {get_object_info(class_name)}")
        return processed_image, "<br>".join(all_info)
    else:
        return processed_image, OBJECT_INFO["no_detection"]


def generate_frames_camera():
    """Generates frames from the camera for live streaming."""
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera.")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) 

# --- Flask Routes ---

@app.route('/')
def index():
    """Main page with camera feed and upload options."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint for live camera feed."""
    return Response(generate_frames_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_detect', methods=['POST'])
def capture_and_detect():
    """Captures a frame from camera, runs detection, and returns results."""
    if not camera.isOpened():
        return jsonify({"error": "Camera not available"}), 500

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image from camera"}), 500

    if session:
        try:
            # Preprocess image
            input_tensor, original_shape = preprocess_image_for_onnx(frame, IMG_SIZE)
            
            # Run ONNX inference
            outputs = session.run(output_names, {input_name: input_tensor})
            
            # Process outputs (NMS, scaling)
            detections = process_detections_from_onnx_output(outputs, original_shape, IMG_SIZE)

            # Draw detections
            annotated_frame, info_text = draw_detections(frame, detections)

            # Encode annotated frame to base64
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                "image": f"data:image/jpeg;base64,{encoded_image}",
                "info": info_text
            })
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            ret, buffer = cv2.imencode('.jpg', frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                "image": f"data:image/jpeg;base64,{encoded_image}",
                "info": f"Error during object detection: {e}"
            })
    else:
        ret, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "info": "Model not loaded. Cannot perform detection."
        })


@app.route('/upload_and_detect', methods=['POST'])
def upload_and_detect():
    """Handles image uploads, runs detection, and returns results."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image file"}), 400

    if file:
        try:
            # Read image using OpenCV from bytes
            filestr = file.read()
            np_array = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if session:
                # Preprocess image
                input_tensor, original_shape = preprocess_image_for_onnx(image, IMG_SIZE)
                
                # Run ONNX inference
                outputs = session.run(output_names, {input_name: input_tensor})
                
                # Process outputs (NMS, scaling)
                detections = process_detections_from_onnx_output(outputs, original_shape, IMG_SIZE)

                # Draw detections
                annotated_image, info_text = draw_detections(image, detections)
                
                ret, buffer = cv2.imencode('.jpg', annotated_image)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    "image": f"data:image/jpeg;base64,{encoded_image}",
                    "info": info_text
                })
            else:
                ret, buffer = cv2.imencode('.jpg', image)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    "image": f"data:image/jpeg;base64,{encoded_image}",
                    "info": "Model not loaded. Cannot perform detection."
                })
        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            return jsonify({"error": f"Failed to process image: {e}"}), 500

if __name__ == '__main__':
    # Run Flask app on all available interfaces (0.0.0.0)
   app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
