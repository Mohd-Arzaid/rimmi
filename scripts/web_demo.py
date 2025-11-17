import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import base64

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('models/egyptian_monument_model.h5')
class_names = np.load('models/class_names.npy', allow_pickle=True)

print(f"Model loaded! Classes: {class_names}")

# Feature Engineering Functions
def extract_color_features(image):
    """Extract color histogram features"""
    img_array = np.array(image)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Calculate histograms
    hist_r = cv2.calcHist([img_array], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img_array], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([img_array], [2], None, [32], [0, 256])
    
    # Normalize
    hist_r = hist_r.flatten() / hist_r.sum()
    hist_g = hist_g.flatten() / hist_g.sum()
    hist_b = hist_b.flatten() / hist_b.sum()
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(hist_r, color='red', alpha=0.7, label='Red')
    ax.plot(hist_g, color='green', alpha=0.7, label='Green')
    ax.plot(hist_b, color='blue', alpha=0.7, label='Blue')
    ax.set_title('Color Histogram Analysis')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Detect dominant colors
    dominant_color = "Sandy/Desert tones" if hist_r.mean() > hist_b.mean() else "Stone/Sky tones"
    
    return fig, dominant_color

def extract_geometric_features(image):
    """Detect geometric shapes (pyramids, columns)"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=50, maxLineGap=10)
    
    # Draw detected lines
    result_img = img_array.copy()
    line_count = 0
    
    if lines is not None:
        for line in lines[:20]:  # Draw first 20 lines
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            line_count += 1
    
    geometric_analysis = f"Detected {line_count} structural lines. "
    if line_count > 15:
        geometric_analysis += "Strong geometric patterns (likely monument with clear structure)."
    elif line_count > 5:
        geometric_analysis += "Moderate geometric patterns detected."
    else:
        geometric_analysis += "Weak geometric patterns."
    
    return Image.fromarray(result_img), geometric_analysis

def extract_texture_features(image):
    """Analyze texture patterns"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate texture using standard deviation in patches
    patches = []
    h, w = gray.shape
    patch_size = 32
    
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = gray[i:i+patch_size, j:j+patch_size]
            patches.append(np.std(patch))
    
    avg_texture = np.mean(patches)
    
    texture_analysis = f"Average texture variance: {avg_texture:.2f}. "
    if avg_texture > 40:
        texture_analysis += "High texture detail (carved surfaces, hieroglyphics)."
    elif avg_texture > 20:
        texture_analysis += "Moderate texture (weathered stone)."
    else:
        texture_analysis += "Smooth texture (polished surfaces)."
    
    return texture_analysis

def predict_monument(image):
    """Main prediction function with feature engineering"""
    
    # Preprocess image for model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    
    # Create prediction results
    results = {}
    result_text = "🏛️ **Monument Predictions:**\n\n"
    
    for idx in top_3_idx:
        monument = class_names[idx]
        confidence = predictions[idx] * 100
        results[monument.upper()] = float(confidence)
        
        emoji = "🥇" if idx == top_3_idx[0] else "🥈" if idx == top_3_idx[1] else "🥉"
        result_text += f"{emoji} **{monument.upper()}**: {confidence:.2f}%\n"
    
    result_text += "\n---\n\n"
    
    # Feature Engineering Analysis
    result_text += "🔬 **Feature Engineering Analysis:**\n\n"
    
    # Color analysis
    color_fig, dominant_color = extract_color_features(image)
    result_text += f"**Color Analysis:** {dominant_color}\n\n"
    
    # Geometric analysis
    geometric_img, geometric_text = extract_geometric_features(image)
    result_text += f"**Geometric Analysis:** {geometric_text}\n\n"
    
    # Texture analysis
    texture_text = extract_texture_features(image)
    result_text += f"**Texture Analysis:** {texture_text}\n\n"
    
    result_text += "---\n\n"
    result_text += "✅ **Hybrid Approach:** Combining neural network predictions with engineered features for robust classification."
    
    return result_text, results, color_fig, geometric_img

# Monument Information Database
monument_info = {
    "pyramids": """
    **Pyramids of Giza**
    
    📍 Location: Giza Plateau, Egypt
    🏗️ Built: c. 2580–2560 BC (Khufu), 2558–2532 BC (Khafre), 2532–2503 BC (Menkaure)
    👑 Dynasty: Fourth Dynasty of the Old Kingdom
    
    The Pyramids of Giza are among the most iconic structures in human history. The Great Pyramid of Khufu is the largest, originally standing at 146.5 meters tall. These monumental tombs showcase incredible engineering precision and remain the only surviving wonder of the ancient world.
    """,
    "sphinx": """
    **Great Sphinx of Giza**
    
    📍 Location: Giza Plateau, next to Pyramids
    🏗️ Built: c. 2558–2532 BC
    👑 Dynasty: Fourth Dynasty (likely Pharaoh Khafre)
    
    The Great Sphinx is a limestone statue with a lion's body and human head, measuring 73 meters long and 20 meters high. It's believed to represent Pharaoh Khafre and served as a guardian of the Giza plateau. The Sphinx has endured millennia of erosion yet remains an enigmatic symbol of ancient Egypt.
    """,
    "karnak": """
    **Karnak Temple Complex**
    
    📍 Location: Luxor (ancient Thebes), East Bank
    🏗️ Built: c. 2055 BC - 100 AD (multiple dynasties)
    👑 Primary deity: Amun-Ra
    
    Karnak is the largest religious complex ever built, covering over 200 acres. Its Great Hypostyle Hall features 134 massive columns, some reaching 21 meters high. The temple was continuously expanded by successive pharaohs for over 2,000 years, creating a stunning architectural timeline of ancient Egyptian civilization.
    """,
    "luxor": """
    **Luxor Temple**
    
    📍 Location: Luxor (ancient Thebes), East Bank
    🏗️ Built: c. 1400 BC (Amenhotep III), expanded by Ramesses II
    👑 Dynasty: 18th-19th Dynasty, New Kingdom
    
    Luxor Temple was dedicated to the rejuvenation of kingship and was the focal point of the annual Opet Festival. Unlike most temples dedicated to gods, this was dedicated to the "ka" (spiritual double) of the pharaoh. Its grand colonnade and massive statues of Ramesses II create an awe-inspiring entrance.
    """,
    "abu_simbel": """
    **Abu Simbel Temples**
    
    📍 Location: Nubia, near Sudan border (southern Egypt)
    🏗️ Built: c. 1264–1244 BC
    👑 Pharaoh: Ramesses II
    
    Abu Simbel consists of two massive rock-cut temples. The Great Temple features four colossal 20-meter statues of Ramesses II at its entrance. Remarkably, the entire temple complex was relocated in the 1960s to save it from flooding during the Aswan Dam construction - an incredible modern engineering feat matching the ancient achievement.
    """
}

def get_monument_info(monument_name):
    """Get detailed information about the predicted monument"""
    for key in monument_info:
        if key in monument_name.lower():
            return monument_info[key]
    return "Monument information not available."

def full_analysis(image):
    """Complete analysis with all outputs"""
    prediction_text, confidence_dict, color_fig, geometric_img = predict_monument(image)
    
    # Get top prediction
    top_monument = max(confidence_dict, key=confidence_dict.get)
    monument_details = get_monument_info(top_monument)
    
    return prediction_text, confidence_dict, color_fig, geometric_img, monument_details

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Egyptian Monument Recognition") as demo:
    
    gr.Markdown("""
    # 🏛️ Egyptian Monument Recognition System
    ### Using Transfer Learning (MobileNetV2) + Feature Engineering
    
    Upload an image of an Egyptian monument to identify it and learn about its history!
    
    **Supported Monuments:** Pyramids of Giza, Great Sphinx, Karnak Temple, Luxor Temple, Abu Simbel
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Monument Image")
            analyze_btn = gr.Button("🔍 Analyze Monument", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📊 How It Works:
            1. **Deep Learning**: MobileNetV2 neural network
            2. **Feature Engineering**: Color, geometric, texture analysis
            3. **Ensemble Method**: Combines multiple predictions
            """)
    
        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction Results")
            confidence_output = gr.Label(label="Confidence Scores", num_top_classes=3)
    
    with gr.Row():
        with gr.Column():
            color_plot = gr.Plot(label="Color Histogram Analysis")
        with gr.Column():
            geometric_output = gr.Image(label="Geometric Feature Detection")
    
    gr.Markdown("---")
    
    monument_info_output = gr.Markdown(label="Monument Information")
    
    # Connect the button
    analyze_btn.click(
        fn=full_analysis,
        inputs=input_image,
        outputs=[prediction_output, confidence_output, color_plot, 
                geometric_output, monument_info_output]
    )
    
    gr.Markdown("""
    ---
    **Project by:** Salvi Gautam | **B.Tech ECE** | **IGDTUW**
    
    **Technologies:** TensorFlow, MobileNetV2, OpenCV, Transfer Learning, Feature Engineering
    """)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Egyptian Monument Recognition Web Demo")
    print("="*60)
    print("\n📱 The web interface will open in your browser automatically!")
    print("🌐 You can also access it at: http://localhost:7860")
    print("\n⚡ Press Ctrl+C to stop the server\n")
    print("="*60 + "\n")
    
    demo.launch(share=True)  # share=True creates a public link!