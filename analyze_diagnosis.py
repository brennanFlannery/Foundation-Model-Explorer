"""Script to analyze and categorize diagnoses from CCDI Hub Participants CSV file."""

from __future__ import annotations

import pandas as pd
from collections import Counter
from typing import Dict, List, Set
import re


def normalize_diagnosis(diagnosis: str) -> str:
    """Normalize diagnosis string by removing extra whitespace."""
    if pd.isna(diagnosis) or diagnosis == "":
        return ""
    return diagnosis.strip()


def split_multiple_diagnoses(diagnosis: str) -> List[str]:
    """Split diagnosis string that may contain multiple diagnoses separated by semicolons."""
    if not diagnosis:
        return []
    # Split by semicolon and clean up each diagnosis
    diagnoses = [d.strip() for d in diagnosis.split(';') if d.strip()]
    return diagnoses


def extract_base_disease(diagnosis: str) -> str:
    """Extract the base disease name, removing modifiers like 'NOS', subtypes, etc."""
    if not diagnosis:
        return ""
    
    # Remove common suffixes and modifiers
    diagnosis = diagnosis.strip()
    
    # Remove "NOS" (Not Otherwise Specified)
    diagnosis = re.sub(r',\s*NOS\s*$', '', diagnosis, flags=re.IGNORECASE)
    diagnosis = re.sub(r'\s+NOS\s*$', '', diagnosis, flags=re.IGNORECASE)
    
    # Remove common modifiers in parentheses
    diagnosis = re.sub(r'\s*\([^)]*\)\s*', '', diagnosis)
    
    # Remove common qualifiers
    qualifiers = [
        r',\s*malignant\s*',
        r',\s*benign\s*',
        r',\s*primary\s*',
        r',\s*secondary\s*',
        r',\s*metastatic\s*',
        r',\s*anaplastic\s*',
        r',\s*undifferentiated\s*',
        r',\s*well\s+differentiated\s*',
        r',\s*poorly\s+differentiated\s*',
    ]
    for qualifier in qualifiers:
        diagnosis = re.sub(qualifier, '', diagnosis, flags=re.IGNORECASE)
    
    return diagnosis.strip()


def categorize_diagnosis(diagnosis: str) -> str:
    """Categorize a diagnosis into a main disease category."""
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
        if 'hepatocellular' in diagnosis_lower or 'hepatocellular' in diagnosis_lower:
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


def main():
    """Main analysis function."""
    csv_file = "CCDI Hub Participants Download 2026-01-22 15-20-01.csv"
    
    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Extract all diagnoses
    all_diagnoses = []
    diagnosis_counts = Counter()
    
    for idx, row in df.iterrows():
        diagnosis = normalize_diagnosis(row['Diagnosis'])
        if diagnosis:
            # Split multiple diagnoses
            diagnoses = split_multiple_diagnoses(diagnosis)
            all_diagnoses.extend(diagnoses)
            for diag in diagnoses:
                diagnosis_counts[diag] += 1
    
    print(f"Total diagnosis entries (including multiple per row): {len(all_diagnoses)}")
    print(f"Unique diagnosis strings: {len(diagnosis_counts)}\n")
    
    # Show top 30 most common diagnoses
    print("=" * 80)
    print("TOP 30 MOST COMMON DIAGNOSES (Original Format)")
    print("=" * 80)
    for diag, count in diagnosis_counts.most_common(30):
        print(f"{count:4d} | {diag}")
    
    # Extract base diseases
    base_diseases = Counter()
    for diag in all_diagnoses:
        base = extract_base_disease(diag)
        if base:
            base_diseases[base] += 1
    
    print("\n" + "=" * 80)
    print("TOP 30 MOST COMMON BASE DISEASES (After Normalization)")
    print("=" * 80)
    for diag, count in base_diseases.most_common(30):
        print(f"{count:4d} | {diag}")
    
    # Categorize diagnoses
    category_counts = Counter()
    diagnosis_to_category = {}
    
    for diag in all_diagnoses:
        category = categorize_diagnosis(diag)
        category_counts[category] += 1
        if diag not in diagnosis_to_category:
            diagnosis_to_category[diag] = category
    
    print("\n" + "=" * 80)
    print("MAIN DISEASE CATEGORIES")
    print("=" * 80)
    for category, count in category_counts.most_common():
        percentage = (count / len(all_diagnoses)) * 100
        print(f"{count:4d} ({percentage:5.1f}%) | {category}")
    
    # Show examples for each category
    print("\n" + "=" * 80)
    print("EXAMPLES OF DIAGNOSES IN EACH CATEGORY")
    print("=" * 80)
    category_examples: Dict[str, Set[str]] = {}
    for diag, category in diagnosis_to_category.items():
        if category not in category_examples:
            category_examples[category] = set()
        category_examples[category].add(diag)
    
    for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        print(f"\n{category} ({category_counts[category]} cases):")
        examples = sorted(list(category_examples[category]))[:10]
        for ex in examples:
            print(f"  - {ex}")
        if len(category_examples[category]) > 10:
            print(f"  ... and {len(category_examples[category]) - 10} more variants")
    
    # Create mapping table
    print("\n" + "=" * 80)
    print("DIAGNOSIS SIMPLIFICATION MAPPING")
    print("=" * 80)
    print("\nThis table shows how original diagnoses map to simplified categories:\n")
    
    # Group by category
    category_groups: Dict[str, List[tuple]] = {}
    for diag, count in diagnosis_counts.items():
        category = categorize_diagnosis(diag)
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append((diag, count))
    
    for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        print(f"\n{category}:")
        # Sort by frequency
        sorted_diags = sorted(category_groups[category], key=lambda x: x[1], reverse=True)
        for diag, count in sorted_diags[:15]:  # Show top 15 per category
            print(f"  {count:4d}x | {diag}")
        if len(sorted_diags) > 15:
            print(f"  ... and {len(sorted_diags) - 15} more variants")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total diagnosis entries: {len(all_diagnoses)}")
    print(f"Unique diagnosis strings: {len(diagnosis_counts)}")
    print(f"Unique base diseases: {len(base_diseases)}")
    print(f"Main categories: {len(category_counts)}")
    print("\nRecommended simplified categories:")
    for category, count in category_counts.most_common():
        percentage = (count / len(all_diagnoses)) * 100
        if percentage >= 1.0:  # Only show categories with >= 1% of cases
            print(f"  - {category}: {count} cases ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
