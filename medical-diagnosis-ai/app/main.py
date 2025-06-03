from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from PIL import Image, ImageStat
import io
import base64
import tensorflow as tf
from transformers import pipeline
import json
from typing import List, Dict, Any
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Diagnosis AI System",
    description="AI-powered medical diagnosis using X-ray/MRI images and medical reports",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageValidator:
    """Validate if uploaded images are actually medical images (X-ray or MRI)"""
    
    def __init__(self):
        pass
    
    def calculate_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate various image features for medical image detection"""
        # Convert to PIL Image for analysis
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Calculate basic statistics
        stat = ImageStat.Stat(pil_image)
        
        features = {
            'mean_intensity': np.mean(stat.mean),
            'std_intensity': np.mean(stat.stddev),
            'aspect_ratio': image.shape[1] / image.shape[0],
            'total_pixels': image.shape[0] * image.shape[1]
        }
        
        # Calculate histogram features
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / np.sum(hist)
        
        # Medical images typically have specific intensity distributions
        features['hist_entropy'] = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features['dark_pixel_ratio'] = np.sum(gray_image < 50) / gray_image.size
        features['bright_pixel_ratio'] = np.sum(gray_image > 200) / gray_image.size
        
        # Edge density (medical images often have specific edge patterns)
        edges = cv2.Canny(gray_image, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def is_likely_xray(self, image: np.ndarray) -> tuple[bool, str, float]:
        """
        Determine if an image is likely an X-ray
        Returns: (is_xray, reason, confidence)
        """
        features = self.calculate_image_features(image)
        
        confidence = 0.0
        reasons = []
        
        # Check if image is grayscale or appears grayscale
        if len(image.shape) == 2:
            confidence += 0.4
            reasons.append("Grayscale format")
        else:
            # Check if color image appears grayscale
            b, g, r = cv2.split(image)
            if np.allclose(b, g, atol=15) and np.allclose(g, r, atol=15):
                confidence += 0.3
                reasons.append("Appears grayscale")
            else:
                confidence -= 0.3  # Penalty for colorful images
                reasons.append("Too colorful for X-ray")
        
        # Check intensity distribution (X-rays typically have bimodal distribution)
        if 30 < features['mean_intensity'] < 120:  # Typical X-ray intensity range
            confidence += 0.25
            reasons.append("Appropriate intensity range")
        else:
            confidence -= 0.2
            reasons.append("Intensity not typical for X-ray")
        
        # Check for high contrast (important for X-rays)
        if features['std_intensity'] > 45:
            confidence += 0.2
            reasons.append("High contrast")
        else:
            confidence -= 0.15
            reasons.append("Low contrast")
        
        # Check aspect ratio (medical images are usually not extremely wide/tall)
        if 0.5 < features['aspect_ratio'] < 2.0:
            confidence += 0.1
            reasons.append("Medical aspect ratio")
        else:
            confidence -= 0.2
            reasons.append("Extreme aspect ratio")
        
        # Check for appropriate edge density
        if 0.02 < features['edge_density'] < 0.15:
            confidence += 0.1
            reasons.append("Medical edge density")
        else:
            confidence -= 0.1
            reasons.append("Edge density not medical-like")
        
        # Check for dark background (common in X-rays)
        if features['dark_pixel_ratio'] > 0.3:
            confidence += 0.1
            reasons.append("Dark background present")
        
        # Size check (medical images are usually of sufficient resolution)
        if features['total_pixels'] > 50000:  
            confidence += 0.05
            reasons.append("Sufficient resolution")
        
        # Additional penalties for clearly non-medical content
        if features['bright_pixel_ratio'] > 0.4:  # Too many bright pixels
            confidence -= 0.3
            reasons.append("Too many bright pixels (sky-like)")
        
        is_xray = confidence > 0.6  # Increased threshold
        reason_text = "; ".join(reasons) if reasons else "No medical image characteristics detected"
        
        return is_xray, reason_text, max(0.0, confidence)
    
    def is_likely_mri(self, image: np.ndarray) -> tuple[bool, str, float]:
        """
        Determine if an image is likely an MRI
        Returns: (is_mri, reason, confidence)
        """
        features = self.calculate_image_features(image)
        
        confidence = 0.0
        reasons = []
        
        # Check if image is grayscale or appears grayscale
        if len(image.shape) == 2:
            confidence += 0.4
            reasons.append("Grayscale format")
        else:
            # Check if color image appears grayscale
            b, g, r = cv2.split(image)
            if np.allclose(b, g, r, atol=15):
                confidence += 0.3
                reasons.append("Appears grayscale")
            else:
                confidence -= 0.4  # Strong penalty for colorful images
                reasons.append("Too colorful for MRI")
        
        # MRIs typically have moderate intensity ranges
        if 40 < features['mean_intensity'] < 150:
            confidence += 0.25
            reasons.append("MRI intensity range")
        else:
            confidence -= 0.2
            reasons.append("Intensity not typical for MRI")
        
        # MRIs have moderate contrast (less than X-rays)
        if 20 < features['std_intensity'] < 70:
            confidence += 0.2
            reasons.append("Moderate contrast")
        else:
            confidence -= 0.15
            reasons.append("Contrast not MRI-like")
        
        # Check aspect ratio
        if 0.7 < features['aspect_ratio'] < 1.5:
            confidence += 0.15
            reasons.append("Square-like aspect ratio")
        else:
            confidence -= 0.2
            reasons.append("Aspect ratio not typical for MRI")
        
        # MRIs typically have lower edge density (smoother)
        if 0.01 < features['edge_density'] < 0.08:
            confidence += 0.15
            reasons.append("Smooth MRI-like edges")
        else:
            confidence -= 0.1
            reasons.append("Edge density not MRI-like")
        
        # MRIs often have more uniform backgrounds
        if 0.1 < features['dark_pixel_ratio'] < 0.5:
            confidence += 0.1
            reasons.append("Uniform background")
        
        # Size check
        if features['total_pixels'] > 50000:
            confidence += 0.05
            reasons.append("Sufficient resolution")
        
        # Additional penalties for clearly non-medical content
        if features['bright_pixel_ratio'] > 0.3:  # Too many bright pixels
            confidence -= 0.3
            reasons.append("Too many bright pixels (outdoor-like)")
        
        is_mri = confidence > 0.6  # Increased threshold
        reason_text = "; ".join(reasons) if reasons else "No MRI characteristics detected"
        
        return is_mri, reason_text, max(0.0, confidence)
    
    def detect_non_medical_features(self, image: np.ndarray) -> tuple[bool, str]:
        """
        Detect if image has characteristics that suggest it's NOT a medical image
        Returns: (is_non_medical, reason)
        """
        features = self.calculate_image_features(image)
        
        non_medical_indicators = []
        
        # Check for very colorful images (medical images are typically grayscale)
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            color_variance = np.var([np.mean(b), np.mean(g), np.mean(r)])
            if color_variance > 300:  # Lowered threshold for better detection
                non_medical_indicators.append("High color variance (nature/object photo)")
            
            # Check for distinct color channels (non-grayscale)
            if not (np.allclose(b, g, atol=20) and np.allclose(g, r, atol=20)):
                # Calculate color intensity differences
                color_diff = np.mean(np.abs(b.astype(float) - g.astype(float))) + \
                            np.mean(np.abs(g.astype(float) - r.astype(float)))
                if color_diff > 30:
                    non_medical_indicators.append("Strong color channels (not medical grayscale)")
        
        # Check for very bright or very dark images
        if features['mean_intensity'] < 25:
            non_medical_indicators.append("Too dark (possibly night photo)")
        elif features['mean_intensity'] > 170:
            non_medical_indicators.append("Too bright (possibly outdoor photo)")
        
        # Check for extreme aspect ratios
        if features['aspect_ratio'] < 0.4 or features['aspect_ratio'] > 2.5:
            non_medical_indicators.append("Extreme aspect ratio (banner/panoramic)")
        
        # Check for very low resolution
        if features['total_pixels'] < 15000:  # Less than ~120x120
            non_medical_indicators.append("Too low resolution for medical imaging")
        
        # Check for very high edge density (suggests detailed photos)
        if features['edge_density'] > 0.2:
            non_medical_indicators.append("Too many details (possibly nature/object photo)")
        
        # Check for outdoor-like characteristics (high bright pixel ratio)
        if features['bright_pixel_ratio'] > 0.5:
            non_medical_indicators.append("Too many bright pixels (outdoor/sky-like)")
        
        # Check for skin-tone like colors (potential selfie)
        if len(image.shape) == 3:
            # Convert to HSV to check for skin tones
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Skin tone range in HSV
            skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 150, 255))
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            if skin_ratio > 0.3:
                non_medical_indicators.append("High skin-tone content (possible selfie)")
        
        # Be more strict - require only 1 indicator for rejection
        is_non_medical = len(non_medical_indicators) >= 1
        reason = "; ".join(non_medical_indicators) if non_medical_indicators else ""
        
        return is_non_medical, reason
    
    def validate_medical_image(self, image: np.ndarray, expected_type: str) -> Dict[str, Any]:
        """
        Validate if the image matches the expected medical image type
        """
        # First check for obvious non-medical characteristics
        is_non_medical, non_medical_reason = self.detect_non_medical_features(image)
        
        if is_non_medical:
            return {
                "is_valid": False,
                "error_type": "non_medical",
                "message": f"This appears to be a non-medical image: {non_medical_reason}",
                "confidence": 0.0,
                "suggestions": [
                    "Please upload an actual medical image",
                    "Ensure the image is an X-ray or MRI scan",
                    "Check that the image shows anatomical structures"
                ]
            }
        
        if expected_type.lower() == "x-ray":
            is_valid, reason, confidence = self.is_likely_xray(image)
            image_type_name = "X-ray"
        elif expected_type.lower() == "mri":
            is_valid, reason, confidence = self.is_likely_mri(image)
            image_type_name = "MRI"
        else:
            return {
                "is_valid": False,
                "error_type": "invalid_type",
                "message": "Invalid image type specified",
                "confidence": 0.0
            }
        
        if is_valid:
            return {
                "is_valid": True,
                "message": f"Image appears to be a valid {image_type_name}",
                "confidence": confidence,
                "detected_features": reason
            }
        else:
            return {
                "is_valid": False,
                "error_type": "wrong_medical_type",
                "message": f"Image doesn't appear to be a {image_type_name}. Detected features: {reason}",
                "confidence": confidence,
                "suggestions": [
                    f"Please upload an actual {image_type_name} image",
                    f"Ensure the image shows {image_type_name} characteristics",
                    "Check if you selected the correct image type",
                    "Try uploading a different medical image"
                ]
            }

# Initialize AI models (mock implementations for demonstration)
class MedicalImageAnalyzer:
    def __init__(self):
        self.model_loaded = False
        self.diseases = [
            "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer", 
            "Fracture", "Arthritis", "Brain Tumor", "Stroke"
        ]
        
    def analyze_xray(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze X-ray image for diseases"""
        # Mock analysis - in real implementation, use trained CNN model
        height, width = image.shape[:2]
        
        # Simulate AI prediction with random but realistic probabilities
        np.random.seed(hash(image.tobytes()) % 2**32)
        
        if np.mean(image) < 100:  # Dark image might indicate pneumonia
            probabilities = [0.85, 0.1, 0.03, 0.02]
        else:
            probabilities = [0.15, 0.05, 0.05, 0.75]
            
        results = {
            "primary_diagnosis": "Pneumonia" if probabilities[0] > 0.5 else "Normal",
            "confidence": max(probabilities),
            "disease_probabilities": {
                "Pneumonia": probabilities[0],
                "COVID-19": probabilities[1], 
                "Tuberculosis": probabilities[2],
                "Normal": probabilities[3]
            },
            "findings": [
                "Opacity in right lower lobe",
                "Possible consolidation pattern",
                "No pleural effusion detected"
            ]
        }
        
        return results
    
    def analyze_mri(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze MRI image for diseases"""
        # Mock MRI analysis
        np.random.seed(hash(image.tobytes()) % 2**32)
        
        probabilities = np.random.dirichlet([1, 1, 1, 10])  # Bias toward normal
        
        results = {
            "primary_diagnosis": "Brain Tumor" if probabilities[0] > 0.3 else "Normal",
            "confidence": max(probabilities),
            "disease_probabilities": {
                "Brain Tumor": probabilities[0],
                "Stroke": probabilities[1],
                "MS Lesions": probabilities[2], 
                "Normal": probabilities[3]
            },
            "findings": [
                "No obvious mass lesions",
                "Normal brain parenchyma",
                "No midline shift"
            ]
        }
        
        return results

class MedicalReportAnalyzer:
    def __init__(self):
        # Initialize NLP pipeline for medical text analysis
        try:
            self.classifier = pipeline("text-classification", 
                                     model="distilbert-base-uncased",
                                     return_all_scores=True)
        except:
            self.classifier = None
        
        # Initialize spaCy model (fallback to basic if scispacy not available)
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                logger.info("Loaded scispaCy medical model")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded standard spaCy model")
                except OSError:
                    self.nlp = None
                    logger.warning("No spaCy model available, using basic text processing")
        except ImportError:
            self.nlp = None
            logger.warning("spaCy not available, using basic text processing")
        
        # Expanded medical vocabularies
        self.medical_keywords = {
            "respiratory_symptoms": [
                "cough", "shortness of breath", "dyspnea", "chest pain", "wheezing",
                "sputum", "hemoptysis", "respiratory distress", "difficulty breathing",
                "tachypnea", "orthopnea", "stridor", "rales", "rhonchi"
            ],
            "cardiovascular_symptoms": [
                "chest pain", "palpitations", "syncope", "edema", "cyanosis",
                "bradycardia", "tachycardia", "murmur", "hypertension", "hypotension"
            ],
            "neurological_symptoms": [
                "headache", "dizziness", "seizure", "weakness", "numbness",
                "confusion", "memory loss", "speech difficulties", "tremor",
                "paralysis", "ataxia", "aphasia", "hemiparesis"
            ],
            "gastrointestinal_symptoms": [
                "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain",
                "bloating", "heartburn", "dysphagia", "hematemesis", "melena"
            ],
            "constitutional_symptoms": [
                "fever", "fatigue", "weight loss", "weight gain", "night sweats",
                "chills", "malaise", "weakness", "anorexia", "lethargy"
            ],
            "infectious_symptoms": [
                "fever", "chills", "sweats", "lymphadenopathy", "erythema",
                "purulent discharge", "abscess", "cellulitis"
            ]
        }
        
        self.medical_conditions = {
            "respiratory_conditions": [
                "pneumonia", "covid-19", "tuberculosis", "lung cancer", "asthma",
                "copd", "bronchitis", "pneumothorax", "pulmonary embolism",
                "lung nodule", "pleural effusion", "atelectasis"
            ],
            "cardiovascular_conditions": [
                "myocardial infarction", "heart failure", "atrial fibrillation",
                "hypertension", "coronary artery disease", "cardiomyopathy",
                "pericarditis", "endocarditis", "valve disease"
            ],
            "neurological_conditions": [
                "stroke", "brain tumor", "seizure disorder", "multiple sclerosis",
                "parkinson's disease", "alzheimer's disease", "migraine",
                "meningitis", "encephalitis", "traumatic brain injury"
            ],
            "musculoskeletal_conditions": [
                "fracture", "arthritis", "osteoporosis", "muscle strain",
                "ligament tear", "disc herniation", "scoliosis", "fibromyalgia"
            ],
            "infectious_conditions": [
                "sepsis", "urinary tract infection", "cellulitis", "meningitis",
                "endocarditis", "osteomyelitis", "abscess"
            ]
        }
        
        # Severity indicators
        self.severity_indicators = {
            "high": [
                "severe", "acute", "critical", "emergent", "life-threatening",
                "unstable", "deteriorating", "worsening rapidly"
            ],
            "moderate": [
                "moderate", "concerning", "persistent", "recurrent",
                "progressive", "ongoing"
            ],
            "mild": [
                "mild", "slight", "minimal", "improving", "stable", "resolved"
            ]
        }
            
    def extract_entities_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using spaCy"""
        if not self.nlp:
            return {"entities": [], "symptoms": [], "conditions": []}
        
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return {"entities": entities}
    
    def find_symptoms_and_conditions(self, text: str) -> Dict[str, Any]:
        """Find symptoms and conditions using expanded dictionaries"""
        text_lower = text.lower()
        
        found_symptoms = {}
        found_conditions = {}
        severity_mentions = []
        
        # Find symptoms by category
        for category, symptoms in self.medical_keywords.items():
            category_symptoms = []
            for symptom in symptoms:
                if symptom in text_lower:
                    category_symptoms.append(symptom)
            if category_symptoms:
                found_symptoms[category] = category_symptoms
        
        # Find conditions by category  
        for category, conditions in self.medical_conditions.items():
            category_conditions = []
            for condition in conditions:
                if condition in text_lower:
                    category_conditions.append(condition)
            if category_conditions:
                found_conditions[category] = category_conditions
        
        # Find severity indicators
        for severity, indicators in self.severity_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    severity_mentions.append({"severity": severity, "term": indicator})
        
        return {
            "symptoms": found_symptoms,
            "conditions": found_conditions,
            "severity_mentions": severity_mentions
        }
    
    def generate_adaptive_recommendations(self, symptoms: Dict, conditions: Dict, severity_mentions: List) -> List[str]:
        """Generate adaptive recommendations based on specific findings"""
        recommendations = []
        
        # Respiratory recommendations
        if "respiratory_symptoms" in symptoms:
            resp_symptoms = symptoms["respiratory_symptoms"]
            
            if "shortness of breath" in resp_symptoms or "dyspnea" in resp_symptoms:
                recommendations.append("Consider pulse oximetry and arterial blood gas analysis")
                recommendations.append("Chest imaging (X-ray or CT) recommended")
            
            if "cough" in resp_symptoms:
                if "sputum" in resp_symptoms:
                    recommendations.append("Sputum culture and sensitivity testing")
                recommendations.append("Consider bronchodilator therapy if indicated")
            
            if "chest pain" in resp_symptoms:
                recommendations.append("ECG and cardiac enzymes to rule out cardiac etiology")
                recommendations.append("Consider pulmonary embolism workup if indicated")
            
            if "hemoptysis" in resp_symptoms:
                recommendations.append("Urgent chest CT and bronchoscopy evaluation")
                recommendations.append("Complete blood count and coagulation studies")
        
        # Constitutional symptoms
        if "constitutional_symptoms" in symptoms:
            const_symptoms = symptoms["constitutional_symptoms"]
            
            if "fever" in const_symptoms:
                recommendations.append("Blood cultures and complete blood count")
                recommendations.append("Monitor for signs of sepsis")
                recommendations.append("Consider infection source evaluation")
            
            if "weight loss" in const_symptoms:
                recommendations.append("Comprehensive metabolic panel and tumor markers")
                recommendations.append("Consider malignancy workup")
            
            if "fatigue" in const_symptoms:
                recommendations.append("Complete blood count and thyroid function tests")
        
        # Cardiovascular recommendations
        if "cardiovascular_symptoms" in symptoms:
            cardio_symptoms = symptoms["cardiovascular_symptoms"]
            
            if "chest pain" in cardio_symptoms:
                recommendations.append("Serial ECGs and cardiac troponins")
                recommendations.append("Consider stress testing or coronary angiography")
            
            if "palpitations" in cardio_symptoms:
                recommendations.append("Holter monitor or event recorder")
                recommendations.append("Electrolyte panel and thyroid function")
            
            if "syncope" in cardio_symptoms:
                recommendations.append("Orthostatic vital signs and ECG")
                recommendations.append("Echocardiogram and neurological evaluation")
        
        # Neurological recommendations  
        if "neurological_symptoms" in symptoms:
            neuro_symptoms = symptoms["neurological_symptoms"]
            
            if "headache" in neuro_symptoms:
                if any(sev["severity"] == "high" for sev in severity_mentions):
                    recommendations.append("Urgent brain imaging (CT or MRI)")
                    recommendations.append("Consider lumbar puncture if indicated")
                else:
                    recommendations.append("Neurological examination and history")
            
            if "seizure" in neuro_symptoms:
                recommendations.append("EEG and brain MRI")
                recommendations.append("Antiepileptic drug levels if applicable")
            
            if "weakness" in neuro_symptoms or "paralysis" in neuro_symptoms:
                recommendations.append("Urgent neurological consultation")
                recommendations.append("Brain and spine imaging")
        
        # Condition-specific recommendations
        if "respiratory_conditions" in conditions:
            resp_conditions = conditions["respiratory_conditions"]
            
            if "pneumonia" in resp_conditions:
                recommendations.append("Antibiotic therapy based on local guidelines")
                recommendations.append("Follow-up chest imaging in 6-8 weeks")
            
            if "covid-19" in resp_conditions:
                recommendations.append("Isolation precautions and contact tracing")
                recommendations.append("Monitor oxygen saturation closely")
        
        # Severity-based recommendations
        high_severity_count = sum(1 for sev in severity_mentions if sev["severity"] == "high")
        if high_severity_count >= 2:
            recommendations.append("URGENT: Consider emergency department evaluation")
            recommendations.append("Continuous monitoring of vital signs")
        
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "Continue supportive care and monitoring",
                "Follow-up with primary care provider in 1-2 weeks",
                "Return if symptoms worsen or new symptoms develop"
            ])
        else:
            recommendations.extend([
                "Patient education regarding warning signs",
                "Ensure medication compliance if applicable",
                "Scheduled follow-up appointment"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def calculate_severity_score(self, symptoms: Dict, conditions: Dict, severity_mentions: List) -> float:
        """Calculate severity score based on findings"""
        base_score = 0.0
        
        # Score based on symptom categories
        symptom_weights = {
            "respiratory_symptoms": 0.3,
            "cardiovascular_symptoms": 0.35,
            "neurological_symptoms": 0.4,
            "constitutional_symptoms": 0.2,
            "gastrointestinal_symptoms": 0.15,
            "infectious_symptoms": 0.25
        }
        
        for category, symptom_list in symptoms.items():
            if category in symptom_weights:
                symptom_count = len(symptom_list)
                base_score += symptom_weights[category] * min(symptom_count / 3.0, 1.0)
        
        # Score based on conditions
        condition_weights = {
            "respiratory_conditions": 0.3,
            "cardiovascular_conditions": 0.4,
            "neurological_conditions": 0.45,
            "infectious_conditions": 0.35,
            "musculoskeletal_conditions": 0.2
        }
        
        for category, condition_list in conditions.items():
            if category in condition_weights:
                condition_count = len(condition_list)
                base_score += condition_weights[category] * min(condition_count / 2.0, 1.0)
        
        # Adjust based on severity mentions
        severity_multiplier = 1.0
        high_severity_count = sum(1 for sev in severity_mentions if sev["severity"] == "high")
        moderate_severity_count = sum(1 for sev in severity_mentions if sev["severity"] == "moderate")
        mild_severity_count = sum(1 for sev in severity_mentions if sev["severity"] == "mild")
        
        if high_severity_count > 0:
            severity_multiplier += 0.3 * high_severity_count
        elif moderate_severity_count > 0:
            severity_multiplier += 0.15 * moderate_severity_count
        elif mild_severity_count > 0:
            severity_multiplier *= 0.8
        
        final_score = min(base_score * severity_multiplier, 1.0)
        return final_score
            
    def analyze_report(self, text: str) -> Dict[str, Any]:
        """Analyze medical report text with enhanced NLP"""
        # Extract entities with spaCy if available
        spacy_entities = self.extract_entities_with_spacy(text)
        
        # Find symptoms and conditions using expanded dictionaries
        findings = self.find_symptoms_and_conditions(text)
        
        # Calculate severity score
        severity_score = self.calculate_severity_score(
            findings["symptoms"], 
            findings["conditions"], 
            findings["severity_mentions"]
        )
        
        # Generate adaptive recommendations
        recommendations = self.generate_adaptive_recommendations(
            findings["symptoms"],
            findings["conditions"], 
            findings["severity_mentions"]
        )
        
        # Flatten symptoms and conditions for summary
        all_symptoms = []
        for category, symptom_list in findings["symptoms"].items():
            all_symptoms.extend(symptom_list)
        
        all_conditions = []
        for category, condition_list in findings["conditions"].items():
            all_conditions.extend(condition_list)
        
        # Generate summary
        if all_symptoms or all_conditions:
            symptom_count = len(all_symptoms)
            condition_count = len(all_conditions)
            
            if condition_count > 0:
                summary = f"Patient presents with {symptom_count} symptoms and {condition_count} suspected conditions. "
            else:
                summary = f"Patient presents with {symptom_count} symptoms. "
            
            if severity_score > 0.7:
                summary += "High-priority case requiring immediate attention."
            elif severity_score > 0.4:
                summary += "Moderate severity requiring close monitoring."
            else:
                summary += "Mild to moderate symptoms requiring routine follow-up."
        else:
            summary = "Limited clinical information available. Recommend comprehensive evaluation."
        
        results = {
            "severity_score": severity_score,
            "key_symptoms": all_symptoms,
            "suspected_conditions": all_conditions,
            "symptom_categories": findings["symptoms"],
            "condition_categories": findings["conditions"],
            "severity_mentions": findings["severity_mentions"],
            "summary": summary,
            "recommendations": recommendations,
            "spacy_entities": spacy_entities.get("entities", []) if spacy_entities else [],
            "analysis_method": "Enhanced NLP with spaCy" if self.nlp else "Basic text processing"
        }
        
        return results

# Initialize analyzers
image_analyzer = MedicalImageAnalyzer()
report_analyzer = MedicalReportAnalyzer()
image_validator = ImageValidator()

@app.get("/")
async def root():
    return {"message": "Medical Diagnosis AI System", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/analyze/image")
async def analyze_medical_image(
    image: UploadFile = File(...),
    image_type: str = Form(...)
):
    """Analyze X-ray or MRI image"""
    try:
        # Read and process image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Validate if the image is actually a medical image of the specified type
        validation_result = image_validator.validate_medical_image(img, image_type)
        
        if not validation_result["is_valid"]:
            error_response = {
                "success": False,
                "error": validation_result["message"],
                "error_type": validation_result["error_type"],
                "confidence": validation_result.get("confidence", 0.0),
                "suggestions": validation_result.get("suggestions", [])
            }
            raise HTTPException(status_code=400, detail=error_response)
        
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        # Resize image for processing
        img_processed = cv2.resize(img_gray, (224, 224))
        
        # Analyze based on image type
        if image_type.lower() == "xray":
            results = image_analyzer.analyze_xray(img_processed)
        elif image_type.lower() == "mri":
            results = image_analyzer.analyze_mri(img_processed)
        else:
            raise HTTPException(status_code=400, detail="Unsupported image type")
        
        # Convert image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', img_processed)
        img_base64 = base64.b64encode(buffer).decode()
        
        return {
            "success": True,
            "image_type": image_type,
            "validation": {
                "message": validation_result["message"],
                "confidence": validation_result["confidence"],
                "detected_features": validation_result.get("detected_features", "")
            },
            "analysis": results,
            "processed_image": f"data:image/jpeg;base64,{img_base64}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/report")
async def analyze_medical_report(report_text: str = Form(...)):
    """Analyze medical report text"""
    try:
        if not report_text.strip():
            raise HTTPException(status_code=400, detail="Report text cannot be empty")
        
        results = report_analyzer.analyze_report(report_text)
        
        return {
            "success": True,
            "analysis": results,
            "text_length": len(report_text)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/combined")
async def combined_analysis(
    image: UploadFile = File(None),
    image_type: str = Form(None),
    report_text: str = Form(None)
):
    """Perform combined analysis of image and report"""
    try:
        results = {"success": True, "analyses": {}}
        
        # Analyze image if provided
        if image and image_type:
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Validate the image
            validation_result = image_validator.validate_medical_image(img, image_type)
            
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": f"Image validation failed: {validation_result['message']}",
                    "suggestions": validation_result.get("suggestions", [])
                }
            
            # Convert to grayscale and resize
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            img_processed = cv2.resize(img_gray, (224, 224))
            
            if image_type.lower() == "xray":
                image_results = image_analyzer.analyze_xray(img_processed)
            else:
                image_results = image_analyzer.analyze_mri(img_processed)
                
            results["analyses"]["image"] = image_results
            results["validation"] = validation_result
        
        # Analyze report if provided
        if report_text and report_text.strip():
            report_results = report_analyzer.analyze_report(report_text)
            results["analyses"]["report"] = report_results
        
        # Generate combined insights
        if "image" in results["analyses"] and "report" in results["analyses"]:
            image_conf = results["analyses"]["image"]["confidence"]
            report_severity = results["analyses"]["report"]["severity_score"]
            
            combined_confidence = (image_conf + report_severity) / 2
            
            results["combined_insights"] = {
                "overall_confidence": combined_confidence,
                "risk_level": "High" if combined_confidence > 0.7 else "Moderate" if combined_confidence > 0.4 else "Low",
                "recommendation": "Immediate medical attention required" if combined_confidence > 0.8 else "Schedule follow-up appointment"
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Combined analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 