import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import *
from utils import *

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


def initialize_analyzer():
    """Initialize the age-gender analyzer"""
    if st.session_state.analyzer is None:
        st.session_state.analyzer = AgeGenderAnalyzer()
    return st.session_state.analyzer


def main():
    """Main Streamlit application"""

    # Title and description
    st.title("👥 Age & Gender Detection System")
    st.markdown("### *Powered by InsightFace + DeepFace AI Models*")
    st.markdown("---")

    # Initialize analyzer
    analyzer = initialize_analyzer()

    if not analyzer.model_loaded:
        st.error("⚠️ AI models failed to load. Please refresh the page or check your installation.")
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Analysis Settings")

        # Detection settings
        st.subheader("🎯 Detection")
        detection_confidence = st.slider(
            "Detection Confidence",
            0.1, 1.0, DETECTION_CONFIDENCE, 0.05,
            help="Higher values = more strict face detection"
        )

        min_face_size = st.slider(
            "Minimum Face Size",
            20, 100, MIN_FACE_SIZE, 5,
            help="Minimum face size in pixels"
        )

        # Display settings
        st.subheader("🎨 Display Options")
        show_dual_results = st.checkbox(
            "Show Both Model Results",
            value=True,
            help="Compare InsightFace vs DeepFace results"
        )

        show_emotion = st.checkbox(
            "Show Emotion Analysis",
            value=True,
            help="Display detected emotions"
        )

        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True,
            help="Display model confidence levels"
        )

        show_processing_time = st.checkbox(
            "Show Processing Time",
            value=ENABLE_PERFORMANCE_MONITORING,
            help="Display analysis performance metrics"
        )

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            enable_gpu = st.checkbox("Enable GPU", value=ENABLE_GPU)
            debug_mode = st.checkbox("Debug Mode", value=DEBUG_MODE)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Analysis", "📊 Statistics", "ℹ️ Information"])

    with tab1:
        handle_image_analysis(analyzer, detection_confidence, min_face_size,
                              show_dual_results, show_emotion, show_confidence,
                              show_processing_time)

    with tab2:
        handle_statistics_view()

    with tab3:
        handle_information_view()


def handle_image_analysis(analyzer, detection_confidence, min_face_size,
                          show_dual_results, show_emotion, show_confidence,
                          show_processing_time):
    """Handle the main image analysis interface"""

    st.header("📸 Face Analysis")

    # Input method selection
    input_method = st.radio(
        "Choose Input Method:",
        ["📤 Upload Image", "📷 Camera Capture"],
        horizontal=True
    )

    uploaded_file = None
    camera_image = None

    # Get image input
    if input_method == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image file",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    else:
        camera_image = st.camera_input("Take a picture for analysis")

    # Process the image
    image_source = uploaded_file or camera_image

    if image_source is not None:
        # Validate image
        is_valid, message = validate_image(image_source)

        if not is_valid:
            st.error(f"❌ {message}")
            return

        # Load and display original image
        image = Image.open(image_source)
        image_array = np.array(image)

        # ✅ CORRECTED: Handle color space properly
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Handle RGBA by converting to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Already RGB from PIL, no conversion needed
            image_rgb = image_array
        elif len(image_array.shape) == 2:
            # Grayscale, convert to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            # Fallback
            image_rgb = image_array

        # Create two columns for before/after
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            display_image = resize_image_for_display(image)
            st.image(display_image, caption="Input Image", use_container_width=True)

            # Show image info
            st.info(f"📐 Dimensions: {image.size[0]} × {image.size[1]} pixels")

        with col2:
            st.subheader("🎯 Analysis Results")

            # Perform analysis
            with st.spinner("🧠 Analyzing faces..."):
                start_time = time.time()

                # Update detection parameters
                analyzer.face_app.det_thresh = detection_confidence

                results = analyzer.analyze_image(image_rgb)

                analysis_time = time.time() - start_time

            if results:
                # Filter by minimum face size
                filtered_results = []
                for result in results:
                    bbox = result['bbox']
                    face_width = bbox[2] - bbox[0]
                    face_height = bbox[3] - bbox[1]

                    if face_width >= min_face_size and face_height >= min_face_size:
                        filtered_results.append(result)

                if filtered_results:
                    # Draw annotations
                    annotated_image = draw_analysis_results(
                        image_rgb.copy(),
                        filtered_results,
                        show_dual_results,
                        show_emotion,
                        show_confidence
                    )

                    # Display annotated image - ✅ CORRECTED: Convert to PIL Image properly
                    annotated_pil = Image.fromarray(annotated_image.astype(np.uint8))
                    display_annotated = resize_image_for_display(annotated_pil)
                    st.image(display_annotated, caption="Analysis Results", use_container_width=True)

                    # Processing time
                    if show_processing_time:
                        st.success(f"⚡ Analysis completed in {format_processing_time(analysis_time)}")

                    # Add to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'results': filtered_results,
                        'image_size': image.size,
                        'processing_time': analysis_time
                    })

                    # Display detailed results
                    display_detailed_results(filtered_results, show_dual_results,
                                             show_emotion, show_confidence, show_processing_time)

                    # Download button
                    if st.button("💾 Download Annotated Image", key="download_btn"):
                        download_annotated_image(annotated_image)

                else:
                    st.warning(f"⚠️ No faces detected above minimum size threshold ({min_face_size}px)")

            else:
                st.info("ℹ️ No faces detected in the image")

    else:
        # Show example or instructions
        st.info("👆 Please upload an image or take a photo to start analysis")

        # Show example results (optional)
        if st.checkbox("Show Example Analysis"):
            show_example_analysis()


def display_detailed_results(results, show_dual_results, show_emotion,
                             show_confidence, show_processing_time):
    """Display detailed analysis results"""

    st.subheader("🔍 Detailed Analysis")

    # Summary statistics
    stats = calculate_statistics(results)

    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("👥 Total Faces", stats['total_faces'])
        with col2:
            st.metric("📊 Average Age", f"{stats['avg_age']:.0f} years" if stats['avg_age'] > 0 else "N/A")
        with col3:
            st.metric("👨 Males", stats['male_count'])
        with col4:
            st.metric("👩 Females", stats['female_count'])

    # Individual face results
    for i, result in enumerate(results, 1):
        with st.expander(f"👤 Face {i} - {result['gender_final']}, {result['age_final']} years", expanded=False):

            # Create columns for organized display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**🎂 Age Analysis**")
                if show_dual_results and result['age_deepface']:
                    st.write(f"• InsightFace: {result['age_insightface']} years")
                    st.write(f"• DeepFace: {result['age_deepface']} years")
                    st.write(f"• **Final: {result['age_final']} years**")
                else:
                    st.write(f"• **Age: {result['age_final']} years**")

                st.write(f"• **Group: {result['age_group']}**")

            with col2:
                st.write("**👤 Gender Analysis**")
                if show_dual_results:
                    st.write(f"• InsightFace: {result['gender_insightface']}")
                    st.write(f"• DeepFace: {result['gender_deepface']}")
                    st.write(f"• **Final: {result['gender_final']}**")
                else:
                    st.write(f"• **Gender: {result['gender_final']}**")

                if show_confidence and result['gender_confidence'] > 0:
                    st.write(f"• Confidence: {result['gender_confidence']:.1%}")

            with col3:
                st.write("**📈 Detection Info**")
                if show_confidence:
                    st.write(f"• Detection: {result['detection_score']:.1%}")

                # Face dimensions
                bbox = result['bbox']
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                st.write(f"• Size: {face_width}×{face_height}px")

                if show_emotion and result['emotion'] != 'Unknown':
                    st.write(f"• **Emotion: {result['emotion']}**")

                if show_processing_time:
                    total_time = result['insightface_time'] + result['deepface_time']
                    st.write(f"• Processing: {format_processing_time(total_time)}")

            # Emotion breakdown (if available)
            if show_emotion and result['emotion_scores']:
                st.write("**😊 Emotion Breakdown**")
                emotion_df = pd.DataFrame([
                    {'Emotion': emotion.title(), 'Score': score}
                    for emotion, score in result['emotion_scores'].items()
                ]).sort_values('Score', ascending=False)

                if not emotion_df.empty:
                    fig = px.bar(emotion_df, x='Emotion', y='Score',
                                 title=f"Emotion Analysis - Face {i}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)


def handle_statistics_view():
    """Handle the statistics dashboard"""

    st.header("📊 Analysis Statistics")

    if not st.session_state.analysis_history:
        st.info("📈 No analysis data yet. Analyze some images to see statistics!")
        return

    # Overall statistics
    st.subheader("📈 Overall Statistics")

    total_analyses = len(st.session_state.analysis_history)
    total_faces = sum(len(entry['results']) for entry in st.session_state.analysis_history)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🖼️ Images Analyzed", total_analyses)
    with col2:
        st.metric("👥 Total Faces", total_faces)
    with col3:
        avg_faces = total_faces / total_analyses if total_analyses > 0 else 0
        st.metric("📊 Avg Faces/Image", f"{avg_faces:.1f}")
    with col4:
        total_time = sum(entry['processing_time'] for entry in st.session_state.analysis_history)
        st.metric("⏱️ Total Processing", format_processing_time(total_time))

    # Age and gender distribution
    all_results = []
    for entry in st.session_state.analysis_history:
        all_results.extend(entry['results'])

    if all_results:
        st.subheader("👥 Demographics Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            ages = [r['age_final'] for r in all_results if r['age_final']]
            if ages:
                fig_age = px.histogram(ages, title="Age Distribution",
                                       nbins=20, labels={'value': 'Age', 'count': 'Count'})
                fig_age.update_layout(height=400)
                st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            # Gender distribution
            genders = [r['gender_final'] for r in all_results]
            gender_counts = pd.Series(genders).value_counts()

            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                                title="Gender Distribution")
            fig_gender.update_layout(height=400)
            st.plotly_chart(fig_gender, use_container_width=True)

        # Age groups distribution
        st.subheader("👶 Age Groups Distribution")
        age_groups = [r['age_group'] for r in all_results]
        age_group_counts = pd.Series(age_groups).value_counts()

        fig_age_groups = px.bar(x=age_group_counts.index, y=age_group_counts.values,
                                title="Distribution by Age Groups",
                                labels={'x': 'Age Group', 'y': 'Count'})
        fig_age_groups.update_layout(height=400)
        st.plotly_chart(fig_age_groups, use_container_width=True)

        # Emotion analysis (if available)
        emotions = []
        for result in all_results:
            if result['emotion'] != 'Unknown':
                emotions.append(result['emotion'])

        if emotions:
            st.subheader("😊 Emotion Analysis")
            emotion_counts = pd.Series(emotions).value_counts()

            fig_emotions = px.bar(x=emotion_counts.index, y=emotion_counts.values,
                                  title="Detected Emotions Distribution",
                                  labels={'x': 'Emotion', 'y': 'Count'})
            fig_emotions.update_layout(height=400)
            st.plotly_chart(fig_emotions, use_container_width=True)

        # Analysis history table
        st.subheader("📋 Analysis History")

        history_data = []
        for i, entry in enumerate(st.session_state.analysis_history[-10:], 1):  # Show last 10
            stats = calculate_statistics(entry['results'])
            history_data.append({
                'Analysis #': len(st.session_state.analysis_history) - 10 + i,
                'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Faces Detected': stats['total_faces'],
                'Avg Age': f"{stats['avg_age']:.0f}" if stats['avg_age'] > 0 else "N/A",
                'Males': stats['male_count'],
                'Females': stats['female_count'],
                'Processing Time': format_processing_time(entry['processing_time'])
            })

        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)

        # Clear history button
        if st.button("🗑️ Clear Analysis History"):
            st.session_state.analysis_history = []
            st.success("✅ Analysis history cleared!")
            st.rerun()


def handle_information_view():
    """Handle the information and help section"""

    st.header("ℹ️ System Information")

    # How it works
    st.subheader("🔬 How It Works")

    st.write("""
    This system uses two powerful AI models to analyze faces in images:

    **🎯 InsightFace:**
    - Fast face detection and basic age/gender estimation
    - Real-time processing capabilities
    - High accuracy face detection

    **🧠 DeepFace:**
    - Advanced age estimation using deep learning
    - Precise gender classification
    - Emotion recognition capabilities

    **🔄 Combined Approach:**
    - Uses both models for comparison and validation
    - Provides confidence scores for reliability assessment
    - Offers dual-model results for enhanced accuracy
    """)

    # Features
    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **🎯 Detection Features:**
        - Multi-face detection
        - Adjustable confidence thresholds
        - Minimum face size filtering
        - Real-time camera support

        **📊 Analysis Features:**
        - Age estimation (0-100 years)
        - Gender classification
        - Age group categorization
        - Emotion recognition
        """)

    with col2:
        st.write("""
        **📈 Performance Features:**
        - Processing time monitoring
        - Confidence scoring
        - Batch analysis support
        - Statistical reporting

        **🎨 Display Features:**
        - Dual-model comparison
        - Interactive visualizations
        - Downloadable results
        - Analysis history
        """)

    # Usage instructions
    st.subheader("📝 Usage Instructions")

    with st.expander("🚀 Getting Started"):
        st.write("""
        1. **Choose Input Method**: Upload an image or use your camera
        2. **Adjust Settings**: Use the sidebar to configure detection parameters
        3. **Analyze**: The system will automatically detect and analyze faces
        4. **Review Results**: Check detailed analysis for each detected face
        5. **View Statistics**: Monitor your analysis history and trends
        """)

    with st.expander("⚙️ Settings Guide"):
        st.write("""
        **Detection Confidence**: Higher values = more strict face detection
        - Low (0.1-0.3): Detects more faces, may include false positives
        - Medium (0.4-0.6): Balanced detection
        - High (0.7-1.0): Only very clear faces, fewer false positives

        **Minimum Face Size**: Filters out very small faces
        - Small (20-40px): Detects distant/small faces
        - Medium (50-70px): Standard detection
        - Large (80-100px): Only large, clear faces

        **Display Options**:
        - Dual Results: Compare both AI models
        - Emotion Analysis: Show detected emotions
        - Confidence Scores: Display model certainty
        """)

    # Technical specifications
    st.subheader("🔧 Technical Specifications")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **🖼️ Supported Formats:**
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - BMP (.bmp)
        - TIFF (.tiff)

        **📐 Image Requirements:**
        - Minimum: 50×50 pixels
        - Maximum: 5000×5000 pixels
        - Recommended: 640×640 pixels
        """)

    with col2:
        st.write("""
        **🎯 Detection Capabilities:**
        - Multiple faces per image
        - Age range: 0-100 years
        - Gender: Male/Female
        - Emotions: 7 basic emotions

        **⚡ Performance:**
        - CPU: 1-3 seconds per image
        - GPU: 0.5-1 seconds per image
        - Memory: ~2GB for models
        """)

    # Model information
    st.subheader("🤖 AI Models Information")

    with st.expander("📚 Model Details"):
        st.write("""
        **InsightFace Model:**
        - Architecture: ResNet-based CNN
        - Training: 5M+ face images
        - Accuracy: ~95% age, ~98% gender
        - Speed: Very fast (~100ms)

        **DeepFace Model:**
        - Architecture: VGG-Face, DeepID
        - Training: Diverse demographic datasets
        - Accuracy: ~97% age, ~99% gender
        - Features: Age, gender, emotion, ethnicity
        """)

    # FAQ
    st.subheader("❓ Frequently Asked Questions")

    with st.expander("Why do I get different results from both models?"):
        st.write("""
        Different AI models may give slightly different results because:
        - They're trained on different datasets
        - They use different algorithms
        - They have different strengths and weaknesses

        This is why we show both results - you can compare and choose the most reliable one!
        """)

    with st.expander("How accurate are the age predictions?"):
        st.write("""
        Age prediction accuracy depends on several factors:
        - Image quality and lighting
        - Face angle and expression
        - Age range (middle ages are most accurate)
        - Individual facial characteristics

        Typical accuracy: ±3-5 years for adults, ±2-3 years for children
        """)

    with st.expander("What if no faces are detected?"):
        st.write("""
        Try these solutions:
        - Lower the detection confidence threshold
        - Ensure faces are clearly visible and well-lit
        - Check that faces are not too small in the image
        - Make sure faces are facing mostly toward the camera
        - Try a different image with better quality
        """)


def show_example_analysis():
    """Show example analysis results"""
    st.subheader("📖 Example Analysis")
    st.write("Here's what a typical analysis looks like:")

    # Create example data
    example_data = {
        'Face 1': {'Age': 28, 'Gender': 'Female', 'Emotion': 'Happy', 'Confidence': '94%'},
        'Face 2': {'Age': 35, 'Gender': 'Male', 'Emotion': 'Neutral', 'Confidence': '91%'},
        'Face 3': {'Age': 12, 'Gender': 'Female', 'Emotion': 'Surprise', 'Confidence': '89%'}
    }

    for face, data in example_data.items():
        with st.expander(f"👤 {face} - {data['Gender']}, {data['Age']} years"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Age:** {data['Age']} years")
            with col2:
                st.write(f"**Gender:** {data['Gender']}")
            with col3:
                st.write(f"**Emotion:** {data['Emotion']}")
            st.write(f"**Detection Confidence:** {data['Confidence']}")


def download_annotated_image(annotated_image):
    """Provide download functionality for annotated images"""
    try:
        # ✅ CORRECTED: Ensure proper format for download
        # Convert RGB to PIL Image
        if isinstance(annotated_image, np.ndarray):
            pil_image = Image.fromarray(annotated_image.astype(np.uint8))
        else:
            pil_image = annotated_image

        # Save to bytes
        import io
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')

        # Create download button
        st.download_button(
            label="💾 Download Annotated Image",
            data=img_buffer.getvalue(),
            file_name=f"age_gender_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            help="Download the image with analysis annotations"
        )

        st.success("✅ Image ready for download!")

    except Exception as e:
        st.error(f"❌ Error preparing download: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Run the main application
    main()