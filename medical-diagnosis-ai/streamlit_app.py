import streamlit as st
import requests
import json
import numpy as np
from PIL import Image
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #34495E;
        margin: 1rem 0;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .diagnosis-box {
        background: #F8F9FA;
        border-left: 5px solid #3498DB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: #D5F4E6;
        border-left: 5px solid #27AE60;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #FDF2E9;
        border-left: 5px solid #E67E22;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .error-box {
        background: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .category-box {
        background: #EBF5FB;
        border-left: 3px solid #5DADE2;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 3px;
    }
    
    .urgent-rec {
        background: #FADBD8;
        border-left: 4px solid #E74C3C;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 3px;
        font-weight: bold;
    }
    
    .standard-rec {
        background: #F8F9FA;
        border-left: 3px solid #85C1E9;
        padding: 0.6rem;
        margin: 0.3rem 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

def check_api_connection():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_probability_chart(probabilities: dict, title: str):
    """Create a probability chart using Plotly"""
    diseases = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create color scale based on probability values
    colors = ['#E74C3C' if p > 0.7 else '#F39C12' if p > 0.4 else '#27AE60' for p in probs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=probs,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Conditions",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=400
    )
    
    return fig

def create_symptom_category_chart(symptom_categories: dict):
    """Create a chart showing symptom categories"""
    if not symptom_categories:
        return None
    
    categories = []
    counts = []
    
    for category, symptoms in symptom_categories.items():
        category_name = category.replace('_', ' ').title()
        categories.append(category_name)
        counts.append(len(symptoms))
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color='#3498DB',
            text=counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Symptoms by Category",
        xaxis_title="Symptom Categories",
        yaxis_title="Number of Symptoms",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=300
    )
    
    return fig

def display_analysis_results(results: dict, analysis_type: str):
    """Display analysis results in a formatted way"""
    if analysis_type == "image":
        analysis = results["analysis"]
        
        # Display validation information if available
        if "validation" in results:
            validation = results["validation"]
            st.markdown('<div class="section-header">✅ Image Validation</div>', unsafe_allow_html=True)
            
            confidence = validation["confidence"]
            if confidence > 0.7:
                st.success(f"🎯 {validation['message']} (Confidence: {confidence:.1%})")
            elif confidence > 0.5:
                st.info(f"✅ {validation['message']} (Confidence: {confidence:.1%})")
            else:
                st.warning(f"⚠️ {validation['message']} (Confidence: {confidence:.1%})")
            
            if validation.get("detected_features"):
                st.write(f"**Detected Features:** {validation['detected_features']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">🔍 Diagnosis Results</div>', unsafe_allow_html=True)
            
            primary_diagnosis = analysis["primary_diagnosis"]
            confidence = analysis["confidence"]
            
            # Display primary diagnosis with color coding
            if confidence > 0.7:
                st.markdown(f'<div class="error-box"><strong>Primary Diagnosis:</strong> {primary_diagnosis}<br><strong>Confidence:</strong> {confidence:.1%}</div>', unsafe_allow_html=True)
            elif confidence > 0.4:
                st.markdown(f'<div class="warning-box"><strong>Primary Diagnosis:</strong> {primary_diagnosis}<br><strong>Confidence:</strong> {confidence:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box"><strong>Primary Diagnosis:</strong> {primary_diagnosis}<br><strong>Confidence:</strong> {confidence:.1%}</div>', unsafe_allow_html=True)
            
            # Display findings
            st.markdown('<div class="section-header">📋 Key Findings</div>', unsafe_allow_html=True)
            for finding in analysis["findings"]:
                st.write(f"• {finding}")
        
        with col2:
            st.markdown('<div class="section-header">📊 Disease Probabilities</div>', unsafe_allow_html=True)
            fig = create_probability_chart(analysis["disease_probabilities"], "Disease Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display processed image if available
            if "processed_image" in results:
                st.markdown('<div class="section-header">🖼️ Processed Image</div>', unsafe_allow_html=True)
                st.image(results["processed_image"], caption="Processed Medical Image", use_column_width=True)
    
    elif analysis_type == "report":
        analysis = results["analysis"]
        
        # Main analysis display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">📝 Report Analysis</div>', unsafe_allow_html=True)
            
            severity = analysis["severity_score"]
            if severity > 0.7:
                st.markdown(f'<div class="error-box"><strong>Severity Score:</strong> {severity:.2f} (High Risk)</div>', unsafe_allow_html=True)
            elif severity > 0.4:
                st.markdown(f'<div class="warning-box"><strong>Severity Score:</strong> {severity:.2f} (Moderate Risk)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box"><strong>Severity Score:</strong> {severity:.2f} (Low Risk)</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="diagnosis-box">{analysis["summary"]}</div>', unsafe_allow_html=True)
            
            # Display analysis method
            if "analysis_method" in analysis:
                st.info(f"🔬 Analysis Method: {analysis['analysis_method']}")
        
        with col2:
            # Create severity visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = severity * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Assessment"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced symptom and condition analysis
        if "symptom_categories" in analysis and analysis["symptom_categories"]:
            st.markdown('<div class="section-header">🩺 Categorized Symptoms</div>', unsafe_allow_html=True)
            
            # Create symptom category chart
            symptom_fig = create_symptom_category_chart(analysis["symptom_categories"])
            if symptom_fig:
                st.plotly_chart(symptom_fig, use_container_width=True)
            
            # Display symptoms by category
            col1, col2 = st.columns([1, 1])
            
            with col1:
                for category, symptoms in analysis["symptom_categories"].items():
                    category_name = category.replace('_', ' ').title()
                    symptoms_text = ", ".join([s.title() for s in symptoms])
                    st.markdown(f'<div class="category-box"><strong>{category_name}:</strong><br>{symptoms_text}</div>', 
                              unsafe_allow_html=True)
            
            with col2:
                # Display conditions if available
                if "condition_categories" in analysis and analysis["condition_categories"]:
                    st.markdown("**🔍 Suspected Conditions by Category:**")
                    for category, conditions in analysis["condition_categories"].items():
                        category_name = category.replace('_', ' ').title()
                        conditions_text = ", ".join([c.title() for c in conditions])
                        st.markdown(f'<div class="category-box"><strong>{category_name}:</strong><br>{conditions_text}</div>', 
                                  unsafe_allow_html=True)
        
        # Severity mentions
        if "severity_mentions" in analysis and analysis["severity_mentions"]:
            st.markdown('<div class="section-header">⚡ Severity Indicators</div>', unsafe_allow_html=True)
            severity_df = pd.DataFrame(analysis["severity_mentions"])
            
            # Group by severity level
            severity_groups = severity_df.groupby('severity')['term'].apply(list).to_dict()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'high' in severity_groups:
                    st.error("**High Severity Terms:**")
                    for term in severity_groups['high']:
                        st.write(f"🔴 {term}")
            
            with col2:
                if 'moderate' in severity_groups:
                    st.warning("**Moderate Severity Terms:**")
                    for term in severity_groups['moderate']:
                        st.write(f"🟡 {term}")
            
            with col3:
                if 'mild' in severity_groups:
                    st.success("**Mild Severity Terms:**")
                    for term in severity_groups['mild']:
                        st.write(f"🟢 {term}")
        
        # Enhanced recommendations display
        st.markdown('<div class="section-header">💡 Adaptive Clinical Recommendations</div>', unsafe_allow_html=True)
        
        # Categorize recommendations
        urgent_keywords = ["URGENT", "Immediate", "Emergency", "Critical", "Acute"]
        urgent_recs = []
        standard_recs = []
        
        for rec in analysis["recommendations"]:
            if any(keyword in rec for keyword in urgent_keywords):
                urgent_recs.append(rec)
            else:
                standard_recs.append(rec)
        
        # Display urgent recommendations first
        if urgent_recs:
            st.markdown("**🚨 Urgent Recommendations:**")
            for rec in urgent_recs:
                st.markdown(f'<div class="urgent-rec">⚠️ {rec}</div>', unsafe_allow_html=True)
        
        # Display standard recommendations
        if standard_recs:
            st.markdown("**📋 Standard Recommendations:**")
            for rec in standard_recs:
                st.markdown(f'<div class="standard-rec">• {rec}</div>', unsafe_allow_html=True)
        
        # Display spaCy entities if available
        if "spacy_entities" in analysis and analysis["spacy_entities"]:
            st.markdown('<div class="section-header">🔍 Medical Entities (NLP)</div>', unsafe_allow_html=True)
            
            # Create a DataFrame for better display
            entities_data = []
            for entity in analysis["spacy_entities"]:
                entities_data.append({
                    "Entity": entity["text"],
                    "Type": entity["label"],
                    "Position": f"{entity['start']}-{entity['end']}"
                })
            
            if entities_data:
                entities_df = pd.DataFrame(entities_data)
                st.dataframe(entities_df, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🏥 AI-Powered Medical Diagnosis System</h1>', unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Cannot connect to the FastAPI backend. Please ensure the server is running on port 8000.")
        st.code("python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    st.success("✅ Connected to AI Backend Successfully")
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🧭 Navigation")
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Report Analysis", "Image Analysis", "Combined Analysis", "About System"]
    )
    
    if analysis_mode == "Image Analysis":
        st.markdown('<div class="section-header">🖼️ Medical Image Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image_type = st.selectbox("Select Image Type", ["MRI","X-ray"])
            uploaded_file = st.file_uploader(
                f"Upload {image_type} Image", 
                type=['png', 'jpg', 'jpeg', 'dicom'],
                help=f"Upload a {image_type.lower()} image for AI analysis"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded {image_type} Image", use_column_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("🤖 AI is analyzing your medical image..."):
                        try:
                            files = {"image": uploaded_file.getvalue()}
                            data = {"image_type": image_type.lower()}
                            
                            response = requests.post(f"{API_URL}/analyze/image", files=files, data=data)
                            
                            if response.status_code == 200:
                                results = response.json()
                                display_analysis_results(results, "image")
                            elif response.status_code == 400:
                                # Handle validation errors
                                error_data = response.json()
                                error_detail = error_data.get("detail", {})
                                
                                if isinstance(error_detail, dict) and "error_type" in error_detail:
                                    # This is a validation error
                                    error_type = error_detail["error_type"]
                                    error_message = error_detail["error"]
                                    suggestions = error_detail.get("suggestions", [])
                                    confidence = error_detail.get("confidence", 0.0)
                                    
                                    if error_type == "non_medical":
                                        st.error(f"❌ **Non-Medical Image Detected**")
                                        st.warning(f"**Issue:** {error_message}")
                                        
                                        if suggestions:
                                            st.markdown("**📝 Suggestions:**")
                                            for suggestion in suggestions:
                                                st.write(f"• {suggestion}")
                                        
                                        st.info("💡 **Tip:** Medical images are typically grayscale X-rays or MRI scans showing anatomical structures like bones, organs, or tissues.")
                                    
                                    elif error_type == "wrong_medical_type":
                                        st.error(f"❌ **Incorrect Medical Image Type**")
                                        st.warning(f"**Issue:** {error_message}")
                                        st.info(f"**Validation Confidence:** {confidence:.1%}")
                                        
                                        if suggestions:
                                            st.markdown("**📝 Suggestions:**")
                                            for suggestion in suggestions:
                                                st.write(f"• {suggestion}")
                                        
                                        # Provide specific guidance based on selected type
                                        if image_type.lower() == "xray":
                                            st.markdown("""
                                            **🩻 X-ray Image Characteristics:**
                                            - Usually grayscale or black & white
                                            - High contrast between bones (bright) and soft tissue (dark)
                                            - Shows skeletal structures, lungs, or chest cavity
                                            - Dark background with bright anatomical features
                                            """)
                                        else:  # MRI
                                            st.markdown("""
                                            **🧠 MRI Image Characteristics:**
                                            - Grayscale medical scan
                                            - Smooth gradients and soft tissue contrast
                                            - Shows brain, joints, or internal organs
                                            - More uniform background than X-rays
                                            - Less sharp edges, more smooth transitions
                                            """)
                                    
                                    else:
                                        st.error(f"❌ **Validation Error:** {error_message}")
                                
                                else:
                                    # Handle other 400 errors
                                    st.error(f"❌ **Upload Error:** {error_detail}")
                            else:
                                st.error(f"❌ **Analysis Failed:** {response.json().get('detail', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"❌ **System Error:** {str(e)}")
                            st.info("Please check your internet connection and try again.")
                else:
                    st.warning("Please upload a medical image to analyze.")
        
        with col2:
            st.markdown("### 📋 Instructions")
            st.info(f"""
            **How to use {image_type} Analysis:**
            
            1. Select the appropriate image type ({image_type})
            2. Upload a clear medical image
            3. Click 'Analyze Image' to get AI diagnosis
            4. Review the results and recommendations
            
            **Supported formats:** PNG, JPG, JPEG, DICOM
            
            **Note:** This is a demonstration system. Always consult healthcare professionals for medical decisions.
            """)
            
            # Add specific guidance for valid medical images
            st.markdown("### ✅ Valid Medical Images")
            if image_type.lower() == "xray":
                st.success("""
                **Valid X-ray characteristics:**
                - Grayscale/black & white medical scans
                - Shows bones, lungs, or chest cavity
                - High contrast (bright bones, dark soft tissue)
                - Dark background with anatomical structures
                - Professional medical imaging quality
                """)
                st.error("""
                **❌ Will be rejected:**
                - Nature photos, selfies, artwork
                - Colorful images or screenshots
                - Non-medical content
                - MRI images (use MRI mode instead)
                """)
            else:  # MRI
                st.success("""
                **Valid MRI characteristics:**
                - Grayscale medical brain/body scans
                - Smooth gradients and soft tissue detail
                - Shows internal organs or brain structure
                - More uniform background than X-rays
                - Professional medical imaging quality
                """)
                st.error("""
                **❌ Will be rejected:**
                - Nature photos, selfies, artwork
                - Colorful images or screenshots
                - Non-medical content
                - X-ray images (use X-ray mode instead)
                """)
            
            st.warning("⚠️ **Image Validation:** The system automatically validates that uploaded images are actually medical images of the specified type.")
    
    elif analysis_mode == "Report Analysis":
        st.markdown('<div class="section-header">📝 Medical Report Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            report_text = st.text_area(
                "Enter Medical Report Text",
                height=300,
                placeholder="Enter patient symptoms, medical history, or clinical notes here...",
                help="Paste or type the medical report text for AI analysis"
            )
            
            # Load sample report button
            if st.button("📄 Load Sample Report"):
                try:
                    with open("sample_data/sample_report.txt", "r") as f:
                        sample_text = f.read()
                        st.text_area("Sample Report Loaded:", value=sample_text, height=200, disabled=True)
                        report_text = sample_text
                except FileNotFoundError:
                    st.warning("Sample report file not found")
            
            if st.button("🔍 Analyze Report", type="primary"):
                if report_text.strip():
                    with st.spinner("🤖 AI is analyzing the medical report..."):
                        try:
                            data = {"report_text": report_text}
                            response = requests.post(f"{API_URL}/analyze/report", data=data)
                            
                            if response.status_code == 200:
                                results = response.json()
                                display_analysis_results(results, "report")
                            else:
                                st.error(f"Analysis failed: {response.json().get('detail', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter medical report text to analyze.")
        
        with col2:
            st.markdown("### 💡 Enhanced Analysis Features")
            st.info("""
            **🆕 New Advanced Features:**
            
            • **Categorized Symptoms**: Respiratory, cardiovascular, neurological, etc.
            • **Adaptive Recommendations**: Specific to identified symptoms
            • **Severity Detection**: Automatic severity level assessment
            • **Medical NLP**: Enhanced entity extraction with spaCy
            • **Clinical Guidelines**: Evidence-based recommendations
            
            **Example Enhanced Report:**
            "Patient presents with severe shortness of breath, persistent cough with blood-tinged sputum, and acute chest pain. Physical exam reveals decreased breath sounds and dullness to percussion in the right lower lobe."
            """)
            
            st.markdown("### 📋 Sample Symptoms to Try")
            st.text("""
            • Respiratory: shortness of breath, hemoptysis, chest pain
            • Cardiac: palpitations, syncope, chest pain  
            • Neurological: severe headache, weakness, seizure
            • Constitutional: fever, weight loss, fatigue
            """)
    
    elif analysis_mode == "Combined Analysis":
        st.markdown('<div class="section-header">🔬 Combined Image & Report Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Upload Medical Image")
            image_type = st.selectbox("Image Type", ["X-ray", "MRI"], key="combined_image_type")
            uploaded_file = st.file_uploader(
                "Upload Medical Image", 
                type=['png', 'jpg', 'jpeg'],
                key="combined_image"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"{image_type} Image", use_column_width=True)
        
        with col2:
            st.markdown("#### 📝 Enter Medical Report")
            report_text = st.text_area(
                "Medical Report Text",
                height=200,
                key="combined_report",
                placeholder="Enter clinical findings, symptoms, or medical history..."
            )
        
        st.markdown("---")
        
        if st.button("🔍 Analyze Both Image & Report", type="primary"):
            if uploaded_file or report_text.strip():
                with st.spinner("🤖 Performing comprehensive AI analysis..."):
                    try:
                        files = {}
                        data = {}
                        
                        if uploaded_file:
                            files["image"] = uploaded_file.getvalue()
                            data["image_type"] = image_type.lower()
                        
                        if report_text.strip():
                            data["report_text"] = report_text
                        
                        response = requests.post(f"{API_URL}/analyze/combined", files=files, data=data)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Display individual analyses
                            if "image" in results["analyses"]:
                                st.markdown("### 🖼️ Image Analysis Results")
                                display_analysis_results({"analysis": results["analyses"]["image"]}, "image")
                            
                            if "report" in results["analyses"]:
                                st.markdown("### 📝 Report Analysis Results")
                                display_analysis_results({"analysis": results["analyses"]["report"]}, "report")
                            
                            # Display combined insights
                            if "combined_insights" in results:
                                st.markdown("### 🎯 Combined AI Insights")
                                insights = results["combined_insights"]
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Overall Confidence", 
                                        f"{insights['overall_confidence']:.1%}",
                                        help="Combined confidence from image and report analysis"
                                    )
                                
                                with col2:
                                    risk_color = "🔴" if insights['risk_level'] == "High" else "🟡" if insights['risk_level'] == "Moderate" else "🟢"
                                    st.metric(
                                        "Risk Level",
                                        f"{risk_color} {insights['risk_level']}",
                                        help="Overall risk assessment"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Recommendation",
                                        insights['recommendation'],
                                        help="AI-generated recommendation"
                                    )
                        elif response.status_code == 400:
                            # Handle validation errors in combined analysis
                            error_data = response.json()
                            
                            if "error" in error_data and "Image validation failed" in error_data["error"]:
                                st.error("❌ **Image Validation Failed**")
                                st.warning(f"**Issue:** {error_data['error']}")
                                
                                if "suggestions" in error_data:
                                    st.markdown("**📝 Suggestions:**")
                                    for suggestion in error_data["suggestions"]:
                                        st.write(f"• {suggestion}")
                                
                                st.info("💡 **Tip:** Please upload a valid medical image that matches the selected type (X-ray or MRI).")
                            else:
                                st.error(f"❌ **Error:** {error_data.get('detail', 'Unknown error')}")
                        
                        else:
                            st.error(f"❌ **Analysis Failed:** {response.json().get('detail', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please provide at least an image or report text for analysis.")
    
    elif analysis_mode == "About System":
        st.markdown('<div class="section-header">ℹ️ About AI Medical Diagnosis System</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 System Overview
            
            This AI-powered medical diagnosis system uses advanced deep learning algorithms to analyze:
            
            - **X-ray Images**: Detect pneumonia, COVID-19, tuberculosis, and other chest conditions
            - **MRI Scans**: Identify brain tumors, strokes, and neurological conditions  
            - **Medical Reports**: Extract symptoms, assess severity, and provide recommendations
            
            ### 🆕 Enhanced Features
            
            **Advanced NLP Analysis:**
            - Categorized symptom detection (respiratory, cardiac, neurological, etc.)
            - Adaptive clinical recommendations based on specific findings
            - Severity level detection and scoring
            - Medical entity extraction with spaCy/scispaCy
            
            **Intelligent Recommendations:**
            - Symptom-specific diagnostic workup suggestions
            - Severity-based urgency levels
            - Evidence-based clinical guidelines
            - Adaptive care pathways
            
            **🆕 Smart Image Validation:**
            - Automatic detection of non-medical images
            - X-ray vs MRI type validation
            - Rejection of nature photos, selfies, and other non-medical content
            - Confidence scoring for medical image classification
            
            ### 🔬 Technology Stack
            
            **Backend (FastAPI):**
            - Deep learning models for image analysis
            - Enhanced NLP with spaCy and scispaCy
            - Advanced medical entity recognition
            - REST API for seamless integration
            
            **Frontend (Streamlit):**
            - Interactive web interface
            - Enhanced visualizations with categorized data
            - Real-time analysis results
            - User-friendly medical dashboard
            
            **AI Models:**
            - Convolutional Neural Networks (CNN) for medical imaging
            - Transformer models for text analysis
            - Medical entity recognition models
            - Adaptive recommendation algorithms
            
            ### ⚠️ Important Disclaimer
            
            This system is designed for **educational and demonstration purposes only**.
            
            - **NOT a replacement** for professional medical diagnosis
            - Always consult qualified healthcare professionals
            - Results should be verified by medical experts
            - Use as a supplementary diagnostic tool only
            """)
        
        with col2:
            st.markdown("### 📊 Enhanced System Statistics")
            
            # Mock enhanced statistics for demonstration
            stats_data = {
                "Metric": [
                    "Images Analyzed", 
                    "Reports Processed", 
                    "Symptom Categories", 
                    "Adaptive Recommendations",
                    "Accuracy Rate", 
                    "Response Time"
                ],
                "Value": [
                    "1,234", 
                    "856", 
                    "6 Categories", 
                    "150+ Rules",
                    "94.2%", 
                    "2.3s"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True)
            
            st.markdown("### 🚀 New Features")
            features = [
                "Enhanced medical NLP with spaCy",
                "Categorized symptom analysis", 
                "Adaptive clinical recommendations",
                "Severity-based risk assessment",
                "Medical entity extraction",
                "Evidence-based guidelines",
                "Interactive symptom visualization"
            ]
            
            for feature in features:
                st.write(f"✅ {feature}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7F8C8D;'>"
        "🏥 AI Medical Diagnosis System | Enhanced with spaCy & Adaptive Recommendations"
        "<br>"
        "⚠️ For Educational Use Only"
        "<br>"
        "Made by <a href='https://github.com/divyesh099'>Divyesh Savaliya</a>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 