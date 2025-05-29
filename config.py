# File and Directory Paths
TEMP_DIR = "temp"
MODELS_DIR = "models"

# Detection Settings
DETECTION_CONFIDENCE = 0.5  # Face detection confidence threshold
MIN_FACE_SIZE = 30  # Minimum face size in pixels

# Age Estimation Settings
AGE_GROUPS = {
    "Child": (0, 12),
    "Teenager": (13, 19),
    "Young Adult": (20, 34),
    "Adult": (35, 54),
    "Senior": (55, 100)
}

# UI Settings
MAX_IMAGE_SIZE = (800, 600)  # Maximum display size for images
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

# Performance Settings
RESIZE_FACTOR = 1.0  # Keep original size for better accuracy
ENABLE_GPU = True  # Use GPU if available

# Colors for annotations (BGR format for OpenCV)
COLORS = {
    'face_box': (0, 255, 0),      # Green for face boxes
    'text_bg': (0, 0, 0),         # Black background for text
    'text': (0, 255, 0),          # Green text
    'male': (255, 0, 0),          # Blue for male
    'female': (255, 0, 255)       # Magenta for female
}

# Font settings for annotations
FONT = {
    'face': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'scale': 0.6,
    'thickness': 2
}

# Streamlit Page Configuration
PAGE_CONFIG = {
    'page_title': 'Age & Gender Detection System',
    'page_icon': '👥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Model Configuration
INSIGHTFACE_CONFIG = {
    'ctx_id': 0,  # GPU ID, -1 for CPU
    'det_size': (640, 640),
    'det_thresh': 0.5
}

DEEPFACE_CONFIG = {
    'actions': ['age', 'gender', 'emotion'],
    'enforce_detection': False,
    'detector_backend': 'opencv',
    'align': True
}

# Error messages
ERROR_MESSAGES = {
    'no_face_found': "No face detected in the image. Please ensure faces are clearly visible.",
    'model_load_error': "Error loading AI models. Please check your installation.",
    'image_processing_error': "Error processing image. Please try a different image.",
    'file_error': "Error reading image file. Please check the file format."
}

# Success messages
SUCCESS_MESSAGES = {
    'analysis_complete': "Face analysis completed successfully!",
    'model_loaded': "AI models loaded successfully!"
}

# Development settings
DEBUG_MODE = False  # Set to True for detailed logging
ENABLE_PERFORMANCE_MONITORING = True  # Monitor processing times
LOG_LEVEL = "INFO"