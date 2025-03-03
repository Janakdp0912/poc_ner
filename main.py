# Import necessary libraries
import streamlit as st
import fitz
from PIL import Image, ImageEnhance
import pytesseract
import pandas as pd
import io, os
from typing import List, Dict
import time, logging
from pathlib import Path
from groq import Groq
import json

# Set up configurations
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
SUPPORTED_IMAGES = ["png", "jpg", "jpeg"]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_file(file):
    """Checks if uploaded file meets size requirements"""
    if file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
        return False
    return True

def extract_text_from_pdf(file):
    """Converts PDF content to text"""
    try:
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise

def extract_text_from_image(file):
    """Uses OCR to extract text from images with enhanced preprocessing"""
    try:
        # Open and preprocess image
        image = Image.open(file)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Configure Tesseract for better accuracy
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
        
        return text.strip()
    except Exception as e:
        logger.error(f"Image extraction error: {str(e)}")
        raise

def extract_entities_llm(text: str) -> List[Dict]:
    """Extract entities and their relationships using LLM model"""
    try:
        API_KEY = "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"
        client = Groq(api_key=API_KEY)

        system_prompt = """You are a medical entity extraction expert. Analyze the given medical text carefully and extract all relevant medical entities and their relationships. Focus on:
        - Medical conditions and diseases
        - Medications and dosages
        - Relationships between conditions and medications
        - Lab results and values
        - Symptoms and signs
        - Procedures and treatments
        - Patient demographics
        - Dates and durations
        
        For each condition, identify any medications and dosages that are used to treat it."""
        
        user_prompt = f"""
        Carefully analyze this medical text and extract all medical entities and their relationships:
        {text}

        Important: 
        - Even if the text has OCR artifacts or scanning issues, try to identify any meaningful medical entities and their relationships.
        - For each condition, list the medications and dosages used to treat it.
        - Include relevance information for vital signs, laboratory results, and dates/duration.
        
        Return entities in JSON format with keys: "entity", "type", "position", "related_entities", and "relevance". Example:
        {{
         "entities": [
            {{"entity": "Hypertension", "type": "MEDICAL_CONDITION", "position": 0, "related_entities": [
                {{"entity": "Lisinopril", "type": "MEDICATION", "position": 10}},
                {{"entity": "10 mg", "type": "MEDICATION_DOSAGE", "position": 20}}
            ], "relevance": "High"}},
            {{"entity": "Blood Pressure", "type": "VITAL_SIGN", "position": 30, "relevance": "Moderate"}},
            {{"entity": "Blood Test", "type": "LAB_RESULT", "position": 40, "relevance": "High"}}
          ]
        }}
        """

        # Log the input text for debugging
        logger.info(f"Processing text (first 500 chars): {text[:500]}...")

        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        # Log the LLM response for debugging
        logger.info(f"LLM Response: {completion.choices[0].message.content}")

        response = completion.choices[0].message.content
        result = json.loads(response)
        
        entities = [{"entity": ent["entity"], "type": ent["type"], "related_entities": ent.get("related_entities", [])} for ent in result["entities"]]
        return entities
        
    except Exception as e:
        logger.error(f"LLM entity extraction error: {str(e)}")
        logger.error(f"Input text: {text[:1000]}...")  # Log input text on error
        raise

def export_entities_to_csv(entities):
    """Converts extracted entities to CSV format"""
    try:
        df = pd.DataFrame(entities)
        return df.to_csv(index=False)
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}")
        raise

def group_similar_entity_types(entity_types):
    """Groups similar entity types together"""
    groups = {
        'MEDICATION': ['MEDICATION', 'MEDICATION_DOSAGE', 'DRUG', 'DOSAGE'],
        'CONDITION': ['MEDICAL_CONDITION', 'CONDITION', 'DISEASE', 'DIAGNOSIS'],
        'LAB': ['LAB_RESULT', 'LAB_VALUE', 'TEST_RESULT', 'LABORATORY'],
        'VITAL': ['VITAL_SIGN', 'VITAL', 'BP', 'BLOOD_PRESSURE', 'HEART_RATE'],
        'PROCEDURE': ['PROCEDURE', 'SURGERY', 'TREATMENT', 'INTERVENTION'],
        'DEMOGRAPHIC': ['AGE', 'GENDER', 'SEX', 'PATIENT_INFO', 'DEMOGRAPHIC'],
        'TEMPORAL': ['DATE', 'DURATION', 'TIME', 'FREQUENCY']
    }
    
    grouped_types = {}
    for type_name in entity_types:
        for group, members in groups.items():
            if any(member in type_name.upper() for member in members):
                if group not in grouped_types:
                    grouped_types[group] = []
                grouped_types[group].append(type_name)
                break
        else:
            # If no group matches, put in OTHER
            if 'OTHER' not in grouped_types:
                grouped_types['OTHER'] = []
            grouped_types['OTHER'].append(type_name)
    
    return grouped_types

def main():
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Medical Report Entity Extraction",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Report Entity Extraction")
    st.write("Upload a medical report (PDF or Image) to extract relevant entities.")

    uploaded_file = st.file_uploader(
        "Upload PDF or Image (Max 5MB)",
        type=["pdf"] + SUPPORTED_IMAGES,
        help="Supported formats: PDF, PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        if not validate_file(uploaded_file):
            return

        try:
            with st.spinner("Processing document..."):
                text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_image(uploaded_file)

                with st.expander("View Extracted Text"):
                    st.text(text)

                with st.spinner("Extracting entities using LLM..."):
                    entities = extract_entities_llm(text)
                
                # Display results
                st.subheader("📊 Extracted Entities")
                if entities:
                    # Create dataframe
                    df = pd.DataFrame(entities)
                    
                    # Group entities by type
                    st.subheader("Grouped Medical Entities")
                    entity_types = df['type'].unique()
                    grouped_types = group_similar_entity_types(entity_types)
                    
                    # Display entities with relationships
                    for group, types in grouped_types.items():
                        if group == 'CONDITION':
                            with st.expander(f"🏥 Medical Conditions and Treatments"):
                                for type_name in types:
                                    condition_entities = df[df['type'] == type_name]
                                    for _, condition in condition_entities.iterrows():
                                        st.markdown(f"**{condition['entity']}**")
                                        related_entities = condition.get('related_entities', [])
                                        for related in related_entities:
                                            if related['type'] == 'MEDICATION':
                                                st.markdown(f"• {related['entity']}")
                                            elif related['type'] == 'MEDICATION_DOSAGE':
                                                st.markdown(f"  - Dosage: {related['entity']}")
                        
                        elif group == 'MEDICATION':
                            # Special handling for medications and dosages
                            medications = df[df['type'].str.contains('MEDICATION', case=False) & 
                                          ~df['type'].str.contains('DOSAGE', case=False)]['entity'].tolist()
                            dosages = df[df['type'].str.contains('DOSAGE', case=False)]['entity'].tolist()
                            
                            with st.expander(f"💊 Medications and Dosages"):
                                if len(medications) == len(dosages):
                                    for med, dose in zip(medications, dosages):
                                        st.markdown(f"• {med} - {dose}")
                                else:
                                    st.markdown("**Medications:**")
                                    for med in medications:
                                        st.markdown(f"• {med}")
                                    if dosages:
                                        st.markdown("\n**Dosages:**")
                                        for dose in dosages:
                                            st.markdown(f"• {dose}")
                        
                        elif group == 'VITAL':
                         with st.expander(f"📊 Vital Signs"):
                            for type_name in types:
                                entities_of_type = df[df['type'] == type_name]
                                for _, entity in entities_of_type.iterrows():
                                    st.markdown(f"• {entity['entity']} (Relevance: {entity.get('relevance', 'Unknown')})")

                        elif group == 'LAB':
                         with st.expander(f"🔬 Laboratory Results"):
                            for type_name in types:
                                entities_of_type = df[df['type'] == type_name]
                                for _, entity in entities_of_type.iterrows():
                                    st.markdown(f"• {entity['entity']} (Relevance: {entity.get('relevance', 'Unknown')})")

                        elif group == 'TEMPORAL':
                         with st.expander(f"📅 Dates and Duration"):
                            for type_name in types:
                                temporal_entities = df[df['type'] == type_name]
                                for _, temporal in temporal_entities.iterrows():
                                    st.markdown(f"• {temporal['entity']} (Relevance: {temporal.get('relevance', 'Unknown')})")
                                    related_entities = temporal.get('related_entities', [])
                                    for related in related_entities:
                                        st.markdown(f"  - Related: {related['entity']} ({related['type']})")
                        
                        elif group == 'PROCEDURE':
                            with st.expander(f"⚕️ Procedures and Treatments"):
                                for type_name in types:
                                    entities_of_type = df[df['type'] == type_name]['entity'].tolist()
                                    for entity in entities_of_type:
                                        st.markdown(f"• {entity}")
                        
                        elif group == 'DEMOGRAPHIC':
                            with st.expander(f"👤 Patient Demographics"):
                                for type_name in types:
                                    entities_of_type = df[df['type'] == type_name]['entity'].tolist()
                                    for entity in entities_of_type:
                                        st.markdown(f"• {entity}")
                        
                        

                        
                        
                        else:
                            with st.expander(f"📝 Other Information"):
                                for type_name in types:
                                    entities_of_type = df[df['type'] == type_name]['entity'].tolist()
                                    for entity in entities_of_type:
                                        st.markdown(f"• {entity}")

                    # Show original table in expander
                    with st.expander("View Complete Table"):
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # Show entity distribution chart
                    st.subheader("Entity Type Distribution")
                    type_counts = df['type'].value_counts()
                    st.bar_chart(type_counts)

                    # Add download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = export_entities_to_csv(entities)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name="extracted_entities.csv",
                            mime="text/csv"
                        )
                    with col2:
                        json_str = df.to_json(orient="records")
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="extracted_entities.json",
                            mime="application/json"
                        )
                else:
                    st.info("No entities found in the document.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Processing error")

    # Add sidebar with information
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.write("""
        This tool extracts medical entities from documents using a Groq LLM model.
        
        **Supported entities:**
        - Age
        - Sex
        - History
        - Clinical Events
        - Symptoms
        - Medications
        - And many more...
        
        **File Requirements:**
        - Maximum size: 5MB
        - Supported formats: PDF, PNG, JPG, JPEG
        """)

# Run the application
if __name__ == "__main__":
    main()
