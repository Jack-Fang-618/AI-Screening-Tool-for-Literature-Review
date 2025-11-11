"""
LLM-Based Intelligent Field Mapper

Uses Grok API to intelligently map heterogeneous database fields to standard schema
Handles PubMed, Scopus, Web of Science, Embase, and custom formats
"""

import logging
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()
logger = logging.getLogger(__name__)


class LLMFieldMapper:
    """Intelligent field mapping using Grok LLM"""
    
    # Standard schema for systematic review data
    STANDARD_SCHEMA = {
        'title': 'Article title',
        'abstract': 'Article abstract/summary',
        'authors': 'Author names (comma-separated)',
        'journal': 'Journal or source name',
        'year': 'Publication year',
        'doi': 'Digital Object Identifier',
        'pmid': 'PubMed ID (if available)',
        'keywords': 'Author keywords or subject terms',
        'publication_type': 'Type of publication (Article, Review, etc.)',
        'url': 'URL or link to article',
        'database_source': 'Original database (PubMed, Scopus, etc.)'
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "grok-4-fast-non-reasoning"):
        """
        Initialize LLM Field Mapper
        
        Args:
            api_key: Grok API key (defaults to env var XAI_API_KEY)
            model: Grok model to use (default: grok-4-fast-non-reasoning)
        """
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        self.model = model.strip()  # Remove any trailing spaces
        self.api_url = "https://api.x.ai/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No Grok API key found. LLM mapping will not work.")
    
    def analyze_and_map_fields(
        self,
        columns: List[str],
        sample_data: Optional[List[Dict]] = None,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Use Grok LLM to analyze columns and map to standard schema
        
        Args:
            columns: List of column names from the dataset
            sample_data: Optional sample rows to help LLM understand content
            filename: Optional filename to help detect database source
            
        Returns:
            {
                'detected_database': str,
                'confidence': float,
                'mappings': [
                    {
                        'original_field': str,
                        'standard_field': str,
                        'confidence': float,
                        'reasoning': str
                    }
                ],
                'unmapped_fields': [str],
                'suggested_schema': Dict[str, str]
            }
        """
        if not self.api_key:
            logger.error("Cannot perform LLM mapping: No API key configured")
            return self._fallback_mapping(columns)
        
        # Build prompt for Grok
        prompt = self._build_mapping_prompt(columns, sample_data, filename)
        
        try:
            # Call Grok API
            response = self._call_grok_api(prompt)
            
            # Parse LLM response
            mapping_result = self._parse_llm_response(response, columns)
            
            logger.info(
                f"✅ LLM mapping complete: {mapping_result['detected_database']} "
                f"(confidence: {mapping_result['confidence']:.2f})"
            )
            
            return mapping_result
            
        except Exception as e:
            logger.error(f"LLM mapping failed: {e}")
            return self._fallback_mapping(columns)
    
    def _build_mapping_prompt(
        self,
        columns: List[str],
        sample_data: Optional[List[Dict]],
        filename: Optional[str]
    ) -> str:
        """Build detailed prompt for Grok to analyze fields"""
        
        prompt = f"""You are an expert in systematic literature review data processing. 

**Task**: Analyze the following dataset columns AND sample data content to map them to a standardized schema for systematic reviews.

**Standard Schema Fields** (target):
{json.dumps(self.STANDARD_SCHEMA, indent=2)}

**Dataset Information**:
"""
        
        if filename:
            prompt += f"- **Filename**: {filename}\n"
        
        prompt += f"- **Number of columns**: {len(columns)}\n"
        prompt += f"- **Column names**: {json.dumps(columns, indent=2)}\n"
        
        if sample_data:
            prompt += f"""
**Sample Data** (first {len(sample_data)} rows - ANALYZE CAREFULLY):
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

⚠️ **CRITICAL**: Look at the ACTUAL DATA VALUES, not just column names! 
- Check if values look like titles, abstracts, author names, years, DOIs, etc.
- A column named "AB" containing long text is likely an abstract
- A column named "AU" with names like "Smith, J." is likely authors
- Numbers like "2023", "2024" indicate publication years
- Strings like "10.1234/..." are DOIs
- PMIDs are numeric IDs (usually 7-8 digits)
"""
        else:
            prompt += "\n**Note**: No sample data provided - mapping based on column names only\n"
        
        prompt += """
**Common Database Field Names**:
- **PubMed**: TI (title), AB (abstract), AU (authors), TA (journal), DP (date), AID (doi), PMID
- **Scopus**: Article Title, Abstract, Authors, Source title, Year, DOI, EID
- **Web of Science**: TI, AB, AU, SO (source), PY (year), DI (doi), UT (accession)
- **Embase**: Title, Abstract, Author, Source, Publication Year, DOI, Embase ID

**Instructions**:
1. **Analyze Sample Data Content**: Look at actual values to understand what each column contains
2. **Detect Database Source**: Based on column naming patterns (TI/AB/AU = PubMed, etc.)
3. **Map Fields**: Match each source column to the most appropriate standard field
4. **Consider Data Types**: 
   - Long text (>100 chars) → likely abstract or title
   - Names with commas/semicolons → likely authors
   - 4-digit numbers → likely year
   - "10.xxxx/..." patterns → DOI
   - 7-8 digit numbers → PMID
5. **Confidence Scoring**: 
   - 0.95-1.0: Perfect match (both name AND content clearly match)
   - 0.80-0.94: Strong match (name or content strongly indicates field)
   - 0.60-0.79: Probable match (reasonable inference from available info)
   - <0.60: Uncertain (use this rarely, only when truly ambiguous)
6. **Reasoning**: Explain what in the data/name led to your mapping decision

**Output Format** (JSON only, no markdown):
{{
    "detected_database": "PubMed|Scopus|Web of Science|Embase|Custom",
    "confidence": 0.95,
    "mappings": [
        {{
            "original_field": "TI",
            "standard_field": "title",
            "confidence": 0.98,
            "reasoning": "Column 'TI' is PubMed's abbreviation for title. Sample values like 'A systematic review of...' confirm these are article titles."
        }},
        {{
            "original_field": "AB",
            "standard_field": "abstract",
            "confidence": 0.97,
            "reasoning": "Column 'AB' is PubMed's abbreviation for abstract. Sample values show long descriptive text (200+ chars) typical of abstracts."
        }}
    ],
    "unmapped_fields": ["NIHMS ID", "PMCID"],
    "notes": "PubMed export detected. All core fields successfully mapped. NIHMS ID and PMCID are database-specific identifiers."
}}

**IMPORTANT**: 
- Return ONLY valid JSON, no markdown code blocks or ```json tags
- Analyze BOTH column names AND actual data values in your reasoning
- Every source column must either be mapped OR listed in unmapped_fields
- Higher confidence when both name and content align
- Include specific observations from sample data in your reasoning
"""
        
        return prompt
    
    def _call_grok_api(self, prompt: str) -> str:
        """Call Grok API with the mapping prompt"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert data scientist specializing in systematic literature reviews. You analyze dataset structures and map fields to standardized schemas. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent mapping
            "max_tokens": 2000
        }
        
        logger.info(f"🤖 Calling Grok API for field mapping...")
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        llm_response = result['choices'][0]['message']['content']
        
        logger.debug(f"LLM Response: {llm_response}")
        
        return llm_response
    
    def _parse_llm_response(self, llm_response: str, columns: List[str]) -> Dict:
        """Parse and validate LLM's JSON response"""
        
        # Remove markdown code blocks if present
        llm_response = llm_response.strip()
        if llm_response.startswith('```'):
            # Extract JSON from markdown code block
            lines = llm_response.split('\n')
            llm_response = '\n'.join(lines[1:-1])  # Remove first and last line
        
        try:
            parsed = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {llm_response}")
            return self._fallback_mapping(columns)
        
        # Validate and format response
        return {
            'detected_database': parsed.get('detected_database', 'Unknown'),
            'confidence': float(parsed.get('confidence', 0.5)),
            'mappings': parsed.get('mappings', []),
            'unmapped_fields': parsed.get('unmapped_fields', []),
            'notes': parsed.get('notes', '')
        }
    
    def _fallback_mapping(self, columns: List[str]) -> Dict:
        """Fallback to simple pattern matching if LLM fails"""
        
        logger.warning("Using fallback pattern-based mapping")
        
        # Simple pattern matching (original logic)
        pattern_map = {
            'title': ['title', 'article title', 'ti', 'paper title'],
            'abstract': ['abstract', 'ab', 'summary', 'description'],
            'authors': ['author', 'authors', 'au', 'creator'],
            'journal': ['journal', 'source', 'publication', 'ta', 'so', 'source title'],
            'year': ['year', 'py', 'publication year', 'date', 'dp'],
            'doi': ['doi', 'di', 'digital object identifier'],
            'pmid': ['pmid', 'pubmed id'],
            'keywords': ['keywords', 'kw', 'subject', 'mesh terms']
        }
        
        mappings = []
        unmapped = []
        
        for col in columns:
            col_lower = col.lower()
            mapped = False
            
            for standard_field, patterns in pattern_map.items():
                if any(pattern in col_lower for pattern in patterns):
                    mappings.append({
                        'original_field': col,
                        'standard_field': standard_field,
                        'confidence': 0.6,
                        'reasoning': f'Pattern match: {col_lower} matches {standard_field}'
                    })
                    mapped = True
                    break
            
            if not mapped:
                unmapped.append(col)
        
        return {
            'detected_database': 'Unknown',
            'confidence': 0.5,
            'mappings': mappings,
            'unmapped_fields': unmapped,
            'notes': 'Fallback pattern-based mapping (LLM unavailable)'
        }
    
    def apply_mapping_to_dataframe(self, df, mapping_result: Dict):
        """
        Apply mapping to DataFrame and return standardized version
        
        Handles duplicate mappings by combining values (keeps first non-empty)
        
        Args:
            df: pandas DataFrame
            mapping_result: Result from analyze_and_map_fields()
            
        Returns:
            Standardized pandas DataFrame with renamed columns
        """
        import pandas as pd
        
        # Create a new DataFrame with standard columns
        df_standardized = pd.DataFrame()
        
        # Group mappings by standard field to handle duplicates
        mappings_by_standard = {}
        for m in mapping_result['mappings']:
            standard_field = m['standard_field']
            if standard_field not in mappings_by_standard:
                mappings_by_standard[standard_field] = []
            mappings_by_standard[standard_field].append(m['original_field'])
        
        # Apply mappings, combining duplicates
        for standard_field, original_fields in mappings_by_standard.items():
            if len(original_fields) == 1:
                # Simple rename
                if original_fields[0] in df.columns:
                    df_standardized[standard_field] = df[original_fields[0]]
            else:
                # Multiple columns map to same standard field - combine them
                logger.info(f"📝 Combining {len(original_fields)} columns into '{standard_field}': {original_fields}")
                
                # Take first non-empty value from each row
                combined_series = None
                for orig_field in original_fields:
                    if orig_field in df.columns:
                        if combined_series is None:
                            combined_series = df[orig_field].fillna('').astype(str)
                        else:
                            # Fill empty values with data from next column
                            mask = (combined_series == '') | (combined_series.isna())
                            combined_series = combined_series.where(~mask, df[orig_field].fillna('').astype(str))
                
                if combined_series is not None:
                    df_standardized[standard_field] = combined_series
        
        # Copy unmapped columns
        mapped_original_fields = [f for fields in mappings_by_standard.values() for f in fields]
        for col in df.columns:
            if col not in mapped_original_fields and col not in df_standardized.columns:
                df_standardized[col] = df[col]
        
        # Add missing standard columns with empty values
        for standard_field in self.STANDARD_SCHEMA.keys():
            if standard_field not in df_standardized.columns:
                df_standardized[standard_field] = ''
        
        # Add metadata columns
        df_standardized['database_source'] = mapping_result['detected_database']
        
        # Reorder columns to match standard schema
        standard_columns = list(self.STANDARD_SCHEMA.keys()) + ['database_source']
        
        # Keep unmapped columns at the end
        extra_columns = [col for col in df_standardized.columns if col not in standard_columns]
        final_columns = standard_columns + extra_columns
        
        df_standardized = df_standardized[final_columns]
        
        logger.info(f"✅ Applied mapping: {len(mappings_by_standard)} standard fields created")
        
        return df_standardized
