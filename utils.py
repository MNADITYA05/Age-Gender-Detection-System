import cv2
import numpy as np
import streamlit as st
from PIL import Image
import insightface
from deepface import DeepFace
import logging
import time
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class AgeGenderAnalyzer:
    """Enhanced Age and Gender Detection System"""

    def __init__(self):
        self.face_app = None
        self.model_loaded = False
        self.load_models()

    def load_models(self):
        """Load InsightFace models"""
        try:
            with st.spinner("Loading AI models..."):
                # Initialize InsightFace
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ENABLE_GPU else [
                    'CPUExecutionProvider']

                self.face_app = insightface.app.FaceAnalysis(providers=providers)
                self.face_app.prepare(
                    ctx_id=INSIGHTFACE_CONFIG['ctx_id'],
                    det_size=INSIGHTFACE_CONFIG['det_size']
                )

                # Test DeepFace by running a small analysis
                test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
                try:
                    DeepFace.analyze(test_img, actions=['age'], enforce_detection=False, silent=True)
                except:
                    pass  # Expected to fail, just checking if library works

                self.model_loaded = True
                logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            st.error(f"Failed to load AI models: {str(e)}")
            self.model_loaded = False

    def extract_faces_insightface(self, image):
        """Extract faces using InsightFace"""
        if not self.model_loaded:
            return []

        try:
            start_time = time.time()
            faces = self.face_app.get(image)
            processing_time = time.time() - start_time

            results = []
            for face in faces:
                # Filter by detection confidence
                if face.det_score < DETECTION_CONFIDENCE:
                    continue

                # Check minimum face size
                bbox = face.bbox.astype(int)
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]

                if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                    continue

                result = {
                    'bbox': bbox,
                    'age': int(face.age),
                    'gender': 'Male' if face.gender == 1 else 'Female',
                    'score': float(face.det_score),
                    'processing_time': processing_time
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in InsightFace analysis: {e}")
            return []

    def extract_faces_deepface(self, image, bbox):
        """Extract detailed analysis using DeepFace"""
        try:
            # Crop face region with some padding
            x1, y1, x2, y2 = bbox
            padding = 20

            # Add padding while staying within image bounds
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            face_crop = image[y1:y2, x1:x2]

            # Ensure minimum size
            if face_crop.shape[0] < 48 or face_crop.shape[1] < 48:
                face_crop = cv2.resize(face_crop, (48, 48))

            # Analyze with DeepFace
            start_time = time.time()
            analysis = DeepFace.analyze(
                face_crop,
                actions=DEEPFACE_CONFIG['actions'],
                enforce_detection=DEEPFACE_CONFIG['enforce_detection'],
                detector_backend=DEEPFACE_CONFIG['detector_backend'],
                align=DEEPFACE_CONFIG['align'],
                silent=True
            )
            processing_time = time.time() - start_time

            # Extract results (DeepFace returns a list)
            if isinstance(analysis, list):
                analysis = analysis[0]

            return {
                'age': int(analysis.get('age', 0)),
                'gender': analysis.get('dominant_gender', 'Unknown'),
                'gender_confidence': analysis.get('gender', {}).get(analysis.get('dominant_gender', ''), 50) / 100.0,
                'emotion': analysis.get('dominant_emotion', 'Unknown'),
                'emotion_scores': analysis.get('emotion', {}),
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Error in DeepFace analysis: {e}")
            return {
                'age': None,
                'gender': None,
                'gender_confidence': 0.0,
                'emotion': 'Unknown',
                'emotion_scores': {},
                'processing_time': 0.0
            }

    def classify_age_group(self, age):
        """Classify age into predefined groups"""
        for group_name, (min_age, max_age) in AGE_GROUPS.items():
            if min_age <= age <= max_age:
                return group_name
        return "Unknown"

    def analyze_image(self, image):
        """Complete image analysis pipeline"""
        if not self.model_loaded:
            st.error("Models not loaded. Please refresh the page.")
            return []

        start_time = time.time()

        # Step 1: Extract faces with InsightFace
        insightface_results = self.extract_faces_insightface(image)

        if not insightface_results:
            return []

        # Step 2: Enhance analysis with DeepFace
        final_results = []

        for face_data in insightface_results:
            # Get enhanced analysis from DeepFace
            deepface_data = self.extract_faces_deepface(image, face_data['bbox'])

            # Combine results
            combined_result = {
                # Face detection info
                'bbox': face_data['bbox'],
                'detection_score': face_data['score'],

                # Age analysis
                'age_insightface': face_data['age'],
                'age_deepface': deepface_data['age'],
                'age_final': deepface_data['age'] if deepface_data['age'] else face_data['age'],

                # Gender analysis
                'gender_insightface': face_data['gender'],
                'gender_deepface': deepface_data['gender'],
                'gender_confidence': deepface_data['gender_confidence'],
                'gender_final': deepface_data['gender'] if deepface_data['gender'] else face_data['gender'],

                # Emotion analysis
                'emotion': deepface_data['emotion'],
                'emotion_scores': deepface_data['emotion_scores'],

                # Performance metrics
                'insightface_time': face_data['processing_time'],
                'deepface_time': deepface_data['processing_time']
            }

            # Add age group classification
            final_age = combined_result['age_final']
            combined_result['age_group'] = self.classify_age_group(final_age)

            final_results.append(combined_result)

        total_time = time.time() - start_time

        if ENABLE_PERFORMANCE_MONITORING:
            logger.info(f"Total analysis time: {total_time:.2f}s for {len(final_results)} faces")

        return final_results


def draw_analysis_results(image, results, show_dual_results=True, show_emotion=True, show_confidence=True):
    """Draw analysis results on image"""
    annotated_image = image.copy()

    for i, result in enumerate(results):
        x1, y1, x2, y2 = result['bbox']

        # Choose color based on gender
        if result['gender_final'].lower() == 'male':
            color = COLORS['male']
        elif result['gender_final'].lower() == 'female':
            color = COLORS['female']
        else:
            color = COLORS['face_box']

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # Draw face number
        cv2.circle(annotated_image, (x1 + 15, y1 + 15), 12, color, -1)
        cv2.putText(annotated_image, str(i + 1), (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Prepare labels
        labels = []

        if show_dual_results and result['age_deepface']:
            labels.append(f"Age: {result['age_insightface']} | {result['age_deepface']}")
            labels.append(f"Gender: {result['gender_insightface']} | {result['gender_deepface']}")
        else:
            labels.append(f"Age: {result['age_final']} ({result['age_group']})")
            labels.append(f"Gender: {result['gender_final']}")

        if show_emotion and result['emotion'] != 'Unknown':
            labels.append(f"Emotion: {result['emotion']}")

        if show_confidence:
            labels.append(f"Detection: {result['detection_score']:.2%}")
            if result['gender_confidence'] > 0:
                labels.append(f"Gender Conf: {result['gender_confidence']:.1%}")

        # Draw labels with background
        y_offset = 0
        for label in labels:
            text_y = y2 + 20 + y_offset

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, FONT['face'], FONT['scale'], FONT['thickness']
            )

            # Draw background rectangle
            cv2.rectangle(annotated_image,
                          (x1, text_y - text_height - 5),
                          (x1 + text_width + 10, text_y + baseline + 5),
                          COLORS['text_bg'], -1)

            # Draw text
            cv2.putText(annotated_image, label, (x1 + 5, text_y),
                        FONT['face'], FONT['scale'], color, FONT['thickness'])

            y_offset += text_height + 10

    return annotated_image


def validate_image(uploaded_file):
    """Validate uploaded image"""
    try:
        if uploaded_file is None:
            return False, "No file uploaded"

        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in SUPPORTED_FORMATS:
            return False, f"Unsupported format. Use: {', '.join(SUPPORTED_FORMATS)}"

        # Try to open the image
        image = Image.open(uploaded_file)

        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image too small. Minimum size: 50x50 pixels"

        if image.size[0] > 5000 or image.size[1] > 5000:
            return False, "Image too large. Maximum size: 5000x5000 pixels"

        # Check if image has valid format
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False, "Invalid image format"

        return True, "Valid image"

    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def resize_image_for_display(image, max_size=MAX_IMAGE_SIZE):
    """Resize image for display while maintaining aspect ratio"""
    try:
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            current_size = (width, height)
        else:
            current_size = image.size
            width, height = current_size

        max_width, max_height = max_size

        # Calculate scaling factor
        scale_factor = min(max_width / width, max_height / height, 1.0)

        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            if isinstance(image, np.ndarray):
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image


def calculate_statistics(results):
    """Calculate statistics from analysis results"""
    if not results:
        return {}

    ages = [r['age_final'] for r in results if r['age_final']]

    stats = {
        'total_faces': len(results),
        'avg_age': np.mean(ages) if ages else 0,
        'min_age': min(ages) if ages else 0,
        'max_age': max(ages) if ages else 0,
        'male_count': sum(1 for r in results if r['gender_final'].lower() == 'male'),
        'female_count': sum(1 for r in results if r['gender_final'].lower() == 'female'),
        'age_groups': {}
    }

    # Count age groups
    for group_name in AGE_GROUPS.keys():
        stats['age_groups'][group_name] = sum(1 for r in results if r['age_group'] == group_name)

    return stats


def format_processing_time(seconds):
    """Format processing time for display"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"