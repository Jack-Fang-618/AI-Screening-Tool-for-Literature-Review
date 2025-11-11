"""
Field Mapper - Cross-Database Field Standardization

Intelligent field mapping for heterogeneous database exports
(PubMed, Scopus, Web of Science, Embase)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


# ===== Data Classes =====

@dataclass
class FieldMappingResult:
    """Result from field mapping operation"""
    detected_database: str
    confidence: float
    mappings: Dict[str, str]
    unmapped_fields: List[str]
    warnings: List[str]


# ===== Database Field Mapping Matrix =====

DATABASE_FIELD_MAPPINGS = {
    'pubmed': {
        'TI': 'title',
        'AB': 'abstract',
        'AU': 'authors',
        'TA': 'journal',
        'DP': 'year',
        'AID': 'doi',
        'FAU': 'authors',
        'JT': 'journal',
        'PY': 'year',
        'MH': 'keywords',
        'OT': 'keywords'
    },
    'scopus': {
        'Article Title': 'title',
        'Title': 'title',
        'Abstract': 'abstract',
        'Authors': 'authors',
        'Author(s)': 'authors',
        'Source title': 'journal',
        'Source Title': 'journal',
        'Year': 'year',
        'DOI': 'doi',
        'Author Keywords': 'keywords',
        'Index Keywords': 'keywords'
    },
    'web_of_science': {
        'TI': 'title',
        'AB': 'abstract',
        'AU': 'authors',
        'SO': 'journal',
        'PY': 'year',
        'DI': 'doi',
        'DE': 'keywords',
        'ID': 'keywords'
    },
    'embase': {
        'Title': 'title',
        'Abstract': 'abstract',
        'Author': 'authors',
        'Authors': 'authors',
        'Source': 'journal',
        'Publication Year': 'year',
        'Year': 'year',
        'DOI': 'doi',
        'Keywords': 'keywords'
    }
}


# Standard field patterns for fuzzy matching
STANDARD_FIELD_PATTERNS = {
    'title': [
        'title', 'article title', 'paper title', 'ti', 'article_title',
        'document title', 'publication title'
    ],
    'abstract': [
        'abstract', 'summary', 'description', 'ab', 'synopsis',
        'article abstract', 'abstract text'
    ],
    'authors': [
        'author', 'authors', 'creator', 'au', 'fau', 'author name',
        'author(s)', 'first author', 'author names'
    ],
    'journal': [
        'journal', 'publication', 'source', 'ta', 'so', 'jt',
        'source title', 'journal title', 'publication name'
    ],
    'year': [
        'year', 'date', 'publication year', 'py', 'dp',
        'publication date', 'pub year', 'pubyear'
    ],
    'doi': [
        'doi', 'digital object identifier', 'di', 'aid',
        'doi number', 'article doi'
    ],
    'keywords': [
        'keywords', 'tags', 'subject', 'kw', 'mh', 'de', 'id', 'ot',
        'author keywords', 'index keywords', 'mesh terms'
    ]
}


class FieldMapper:
    """Intelligent field mapper for cross-database standardization"""
    
    def __init__(self):
        self.database_mappings = DATABASE_FIELD_MAPPINGS
        self.field_patterns = STANDARD_FIELD_PATTERNS
    
    def detect_database_source(self, columns: List[str]) -> Tuple[str, float]:
        """
        Detect which database the file came from
        
        Args:
            columns: List of column names from file
            
        Returns:
            Tuple of (database_name, confidence_score)
        """
        columns_lower = [c.lower() for c in columns]
        columns_set = set(columns)
        
        scores = {}
        
        for db_name, mapping in self.database_mappings.items():
            # Count how many database-specific fields match
            matches = sum(1 for field in mapping.keys() if field in columns_set)
            total_db_fields = len(mapping)
            
            # Calculate confidence score
            if total_db_fields > 0:
                scores[db_name] = matches / total_db_fields
        
        if not scores:
            return 'unknown', 0.0
        
        # Return database with highest score
        best_db = max(scores.items(), key=lambda x: x[1])
        return best_db[0], best_db[1]
    
    def auto_map_fields(
        self,
        columns: List[str],
        source: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Auto-detect and map fields to standard schema
        
        Args:
            columns: List of column names from file
            source: Optional database source (if known)
            
        Returns:
            Dict mapping original_field -> standard_field
        """
        mapping = {}
        
        # If source provided, use database-specific mapping
        if source and source in self.database_mappings:
            db_mapping = self.database_mappings[source]
            for col in columns:
                if col in db_mapping:
                    mapping[col] = db_mapping[col]
        
        # For unmapped columns, try fuzzy matching
        for col in columns:
            if col not in mapping:
                best_match, confidence = self._fuzzy_match_field(col)
                if confidence >= 0.7:  # Threshold for auto-mapping
                    mapping[col] = best_match
        
        return mapping
    
    def _fuzzy_match_field(self, column: str) -> Tuple[str, float]:
        """
        Fuzzy match a column name to standard field
        
        Args:
            column: Column name to match
            
        Returns:
            Tuple of (standard_field, confidence_score)
        """
        column_lower = column.lower().strip()
        
        best_match = None
        best_score = 0.0
        
        for standard_field, patterns in self.field_patterns.items():
            for pattern in patterns:
                # Calculate similarity ratio
                ratio = SequenceMatcher(None, column_lower, pattern).ratio()
                
                # Check if exact match (case-insensitive)
                if column_lower == pattern:
                    return standard_field, 1.0
                
                # Update best match if better score
                if ratio > best_score:
                    best_score = ratio
                    best_match = standard_field
        
        return best_match or 'unknown', best_score
    
    def suggest_mapping(self, column: str, target_field: str) -> float:
        """
        Calculate confidence for mapping a column to target field
        
        Args:
            column: Source column name
            target_field: Target standard field name
            
        Returns:
            Confidence score (0-1)
        """
        if target_field not in self.field_patterns:
            return 0.0
        
        column_lower = column.lower().strip()
        patterns = self.field_patterns[target_field]
        
        # Check for exact match
        if column_lower in patterns:
            return 1.0
        
        # Calculate fuzzy match score
        max_ratio = max(
            SequenceMatcher(None, column_lower, pattern).ratio()
            for pattern in patterns
        )
        
        return max_ratio
    
    def validate_mapping(self, mapping: Dict[str, str]) -> List[str]:
        """
        Validate field mapping and return warnings
        
        Args:
            mapping: Dict of original_field -> standard_field
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check if required fields are mapped
        required_fields = ['title', 'abstract']
        mapped_standard_fields = set(mapping.values())
        
        for field in required_fields:
            if field not in mapped_standard_fields:
                warnings.append(f"Required field '{field}' is not mapped")
        
        # Check for duplicate mappings
        from collections import Counter
        field_counts = Counter(mapping.values())
        for field, count in field_counts.items():
            if count > 1:
                warnings.append(
                    f"Standard field '{field}' is mapped from {count} different columns"
                )
        
        return warnings
    
    def merge_with_standard_schema(
        self,
        df: pd.DataFrame,
        mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Apply field mapping and create standardized DataFrame
        
        Args:
            df: Original DataFrame
            mapping: Field mapping dict
            
        Returns:
            Standardized DataFrame
        """
        # Create new DataFrame with standard columns
        standardized = pd.DataFrame()
        
        # Apply mappings
        for original_col, standard_col in mapping.items():
            if original_col in df.columns:
                standardized[standard_col] = df[original_col]
        
        # Add unmapped columns with prefix
        for col in df.columns:
            if col not in mapping:
                standardized[f'custom_{col}'] = df[col]
        
        return standardized
    
    @staticmethod
    def get_required_fields() -> List[str]:
        """Get list of required standard fields"""
        return ['title', 'abstract']
    
    @staticmethod
    def get_optional_fields() -> List[str]:
        """Get list of optional standard fields"""
        return ['authors', 'journal', 'year', 'doi', 'keywords']
