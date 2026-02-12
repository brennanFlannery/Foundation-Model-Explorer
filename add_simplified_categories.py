"""Script to add a simplified diagnosis category column to the CCDI Hub Participants CSV file."""

from __future__ import annotations

import pandas as pd
import re
from typing import List


def categorize_diagnosis(diagnosis: str) -> str:
    """Categorize a diagnosis into a main disease category."""
    if pd.isna(diagnosis) or diagnosis == "":
        return "Unknown"
    
    diagnosis_lower = diagnosis.lower()
    
    # Sarcomas
    if any(term in diagnosis_lower for term in ['sarcoma', 'rhabdomyosarcoma', 'fibrosarcoma']):
        if 'rhabdomyosarcoma' in diagnosis_lower:
            return 'Rhabdomyosarcoma'
        elif 'osteosarcoma' in diagnosis_lower:
            return 'Osteosarcoma'
        elif 'ewing' in diagnosis_lower:
            return 'Ewing Sarcoma'
        elif 'synovial' in diagnosis_lower:
            return 'Synovial Sarcoma'
        elif 'fibrosarcoma' in diagnosis_lower:
            return 'Fibrosarcoma'
        else:
            return 'Sarcoma (Other)'
    
    # Carcinomas
    if 'carcinoma' in diagnosis_lower:
        if 'hepatocellular' in diagnosis_lower:
            return 'Hepatocellular Carcinoma'
        elif 'adrenal' in diagnosis_lower:
            return 'Adrenal Cortical Carcinoma'
        elif 'squamous' in diagnosis_lower:
            return 'Squamous Cell Carcinoma'
        else:
            return 'Carcinoma (Other)'
    
    # Brain/CNS tumors
    if any(term in diagnosis_lower for term in ['medulloblastoma', 'astrocytoma', 'ependymoma', 'glioma']):
        if 'medulloblastoma' in diagnosis_lower:
            return 'Medulloblastoma'
        elif 'astrocytoma' in diagnosis_lower:
            return 'Astrocytoma'
        elif 'ependymoma' in diagnosis_lower:
            return 'Ependymoma'
        else:
            return 'Glioma (Other)'
    
    # Liver tumors
    if 'hepatoblastoma' in diagnosis_lower:
        return 'Hepatoblastoma'
    
    # Kidney tumors
    if 'nephroblastoma' in diagnosis_lower or 'wilms' in diagnosis_lower:
        return 'Nephroblastoma (Wilms Tumor)'
    
    # Neuroblastoma
    if 'neuroblastoma' in diagnosis_lower:
        return 'Neuroblastoma'
    
    # Germ cell tumors
    if any(term in diagnosis_lower for term in ['teratoma', 'yolk sac', 'germ cell']):
        return 'Germ Cell Tumor'
    
    # Nerve sheath tumors
    if 'nerve sheath' in diagnosis_lower:
        return 'Malignant Peripheral Nerve Sheath Tumor'
    
    # Melanoma
    if 'melanoma' in diagnosis_lower:
        return 'Melanoma'
    
    # Other specific tumors
    if 'paraganglioma' in diagnosis_lower:
        return 'Paraganglioma'
    
    # Embryonal sarcoma
    if 'embryonal sarcoma' in diagnosis_lower:
        return 'Embryonal Sarcoma'
    
    return 'Other/Unspecified'


def get_primary_category(diagnosis: str) -> str:
    """Get the primary category for a diagnosis string (handles multiple diagnoses)."""
    if pd.isna(diagnosis) or diagnosis == "":
        return "Unknown"
    
    # Split by semicolon if multiple diagnoses
    diagnoses = [d.strip() for d in diagnosis.split(';') if d.strip()]
    
    if not diagnoses:
        return "Unknown"
    
    # Categorize each diagnosis
    categories = [categorize_diagnosis(d) for d in diagnoses]
    
    # Return the first (primary) category
    # Could be modified to return the most specific or most serious diagnosis
    return categories[0]


def main():
    """Add simplified category column to CSV file."""
    input_file = "CCDI Hub Participants Download 2026-01-22 15-20-01.csv"
    output_file = "CCDI Hub Participants Download 2026-01-22 15-20-01_with_categories.csv"
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Processing {len(df)} rows...")
    
    # Add simplified category column
    df['Diagnosis Category Simplified'] = df['Diagnosis'].apply(get_primary_category)
    
    # Show summary
    print("\nCategory distribution:")
    category_counts = df['Diagnosis Category Simplified'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Save to new file
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Done! New file saved with simplified categories.")
    print(f"\nNote: For participants with multiple diagnoses, the first diagnosis is used for categorization.")


if __name__ == "__main__":
    main()
