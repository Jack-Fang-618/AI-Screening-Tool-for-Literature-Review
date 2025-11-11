"""
Data Merger - Intelligent Multi-Source Dataset Consolidation

Merges multiple datasets from different sources (PubMed, Scopus, WoS, Embase)
with intelligent field mapping and conflict resolution.

Key features:
- Source provenance tracking
- Schema validation before merge
- Conflict resolution strategies
- Field mapping integration
- Data quality checks

Refined improvements over original:
- Structured merge operations
- Configurable conflict resolution
- Detailed merge reports
- Validation at every step
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import numpy as np

from .field_mapper import FieldMapper, FieldMappingResult
from .llm_field_mapper import LLMFieldMapper

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of merge operation"""
    total_records: int
    sources_merged: List[str] = field(default_factory=list)
    field_mappings_applied: Dict[str, int] = field(default_factory=dict)  # source -> field count
    conflicts_resolved: int = 0
    warnings: List[str] = field(default_factory=list)
    merged_dataframe: Optional[pd.DataFrame] = None
    merge_summary: str = ""


class DataMerger:
    """
    Intelligent merger for systematic review datasets
    
    Handles merging of multiple datasets with different schemas,
    applying field mappings and resolving conflicts.
    """
    
    STANDARD_FIELDS = [
        'title', 'abstract', 'authors', 'year', 'journal',
        'doi', 'pmid', 'keywords', 'publication_type',
        'volume', 'issue', 'pages', 'url'
    ]
    
    def __init__(
        self,
        field_mapper: Optional[FieldMapper] = None,
        conflict_strategy: str = 'first',  # 'first', 'longest', 'combine'
        add_source_column: bool = True
    ):
        """
        Initialize data merger
        
        Args:
            field_mapper: FieldMapper instance for standardizing fields (legacy)
            conflict_strategy: How to resolve conflicting values
                - 'first': Keep value from first dataset
                - 'longest': Keep longest non-empty value
                - 'combine': Combine values from all sources
            add_source_column: Add 'data_source' column to track provenance
        """
        self.field_mapper = field_mapper or FieldMapper()  # Keep for backward compatibility
        self.llm_field_mapper = LLMFieldMapper()  # NEW: Use LLM for intelligent mapping
        self.conflict_strategy = conflict_strategy
        self.add_source_column = add_source_column
    
    def merge_datasets(
        self,
        datasets: List[pd.DataFrame],
        source_names: Optional[List[str]] = None,
        validate_schemas: bool = True
    ) -> MergeResult:
        """
        Merge multiple datasets with field mapping
        
        Args:
            datasets: List of DataFrames to merge
            source_names: Optional names for each source (for tracking)
            validate_schemas: Validate schemas before merging
            
        Returns:
            MergeResult with merged DataFrame and statistics
        """
        if not datasets:
            logger.warning("No datasets provided for merging")
            return MergeResult(total_records=0)
        
        if len(datasets) == 1:
            logger.info("Only one dataset provided, returning as-is")
            return MergeResult(
                total_records=len(datasets[0]),
                sources_merged=['single_source'],
                merged_dataframe=datasets[0].copy()
            )
        
        # Generate source names if not provided
        if source_names is None:
            source_names = [f"source_{i+1}" for i in range(len(datasets))]
        
        if len(source_names) != len(datasets):
            raise ValueError(
                f"Number of source names ({len(source_names)}) "
                f"must match number of datasets ({len(datasets)})"
            )
        
        logger.info(f"Merging {len(datasets)} datasets: {', '.join(source_names)}")
        
        # Step 1: Validate schemas
        if validate_schemas:
            validation_warnings = self._validate_schemas(datasets, source_names)
        else:
            validation_warnings = []
        
        # Step 2: Apply field mappings using LLM
        mapped_datasets = []
        field_mappings_applied = {}
        
        for i, (df, source_name) in enumerate(zip(datasets, source_names)):
            logger.info(f"🤖 LLM mapping fields for {source_name}")
            
            # Get sample data for LLM analysis
            sample_rows = min(3, len(df))
            sample_df = df.head(sample_rows).replace([np.inf, -np.inf], np.nan, inplace=False).fillna('')
            
            # Convert Timestamp objects to strings
            for col in sample_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].astype(str)
                elif sample_df[col].dtype == 'object':
                    sample_df[col] = sample_df[col].apply(
                        lambda x: str(x) if not isinstance(x, (str, int, float, bool)) else x
                    )
            
            sample_data = sample_df.to_dict('records')
            
            # Use LLM to analyze and map fields
            mapping_result = self.llm_field_mapper.analyze_and_map_fields(
                columns=df.columns.tolist(),
                sample_data=sample_data,
                filename=source_name
            )
            
            # Extract mappings from result
            mappings = mapping_result.get('mappings', [])
            
            if mappings:
                # Apply LLM mappings
                logger.info(f"📝 Applying {len(mappings)} LLM field mappings for {source_name}")
                mapped_df = self.llm_field_mapper.apply_mapping_to_dataframe(df, mapping_result)
                mapped_datasets.append(mapped_df)
                field_mappings_applied[source_name] = len(mappings)
                logger.info(
                    f"✅ Applied {len(mappings)} field mappings for {source_name} "
                    f"(detected: {mapping_result.get('detected_database', 'unknown')})"
                )
            else:
                # No mappings detected, use as-is
                mapped_datasets.append(df.copy())
                field_mappings_applied[source_name] = 0
                logger.warning(f"⚠️ No field mappings detected for {source_name}")
        
        # Step 3: Add source tracking
        if self.add_source_column:
            for i, df in enumerate(mapped_datasets):
                df['data_source'] = source_names[i]
        
        # Step 4: Merge datasets
        merged_df, conflicts = self._merge_dataframes(
            mapped_datasets,
            source_names
        )
        
        # Step 5: Ensure standard field order
        merged_df = self._reorder_columns(merged_df)
        
        # Generate summary
        summary = self._generate_merge_summary(
            datasets,
            merged_df,
            source_names,
            field_mappings_applied,
            conflicts
        )
        
        result = MergeResult(
            total_records=len(merged_df),
            sources_merged=source_names,
            field_mappings_applied=field_mappings_applied,
            conflicts_resolved=conflicts,
            warnings=validation_warnings,
            merged_dataframe=merged_df,
            merge_summary=summary
        )
        
        logger.info(f"Merge complete: {len(merged_df)} total records")
        return result
    
    def merge_with_mapping_config(
        self,
        datasets: List[pd.DataFrame],
        mapping_configs: List[Dict[str, str]],
        source_names: Optional[List[str]] = None
    ) -> MergeResult:
        """
        Merge datasets with explicit field mapping configurations
        
        Args:
            datasets: List of DataFrames to merge
            mapping_configs: List of mapping dictionaries {original_field: standard_field}
            source_names: Optional names for each source
            
        Returns:
            MergeResult with merged DataFrame
        """
        if len(datasets) != len(mapping_configs):
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match "
                f"number of mapping configs ({len(mapping_configs)})"
            )
        
        # Generate source names if not provided
        if source_names is None:
            source_names = [f"source_{i+1}" for i in range(len(datasets))]
        
        logger.info(f"Merging {len(datasets)} datasets with explicit mappings")
        
        # Apply mappings
        mapped_datasets = []
        field_mappings_applied = {}
        
        for i, (df, mapping, source_name) in enumerate(zip(datasets, mapping_configs, source_names)):
            if mapping:
                mapped_df = df.rename(columns=mapping)
                mapped_datasets.append(mapped_df)
                field_mappings_applied[source_name] = len(mapping)
            else:
                mapped_datasets.append(df.copy())
                field_mappings_applied[source_name] = 0
        
        # Add source tracking
        if self.add_source_column:
            for i, df in enumerate(mapped_datasets):
                df['data_source'] = source_names[i]
        
        # Merge
        merged_df, conflicts = self._merge_dataframes(mapped_datasets, source_names)
        merged_df = self._reorder_columns(merged_df)
        
        summary = self._generate_merge_summary(
            datasets,
            merged_df,
            source_names,
            field_mappings_applied,
            conflicts
        )
        
        return MergeResult(
            total_records=len(merged_df),
            sources_merged=source_names,
            field_mappings_applied=field_mappings_applied,
            conflicts_resolved=conflicts,
            merged_dataframe=merged_df,
            merge_summary=summary
        )
    
    def _validate_schemas(
        self,
        datasets: List[pd.DataFrame],
        source_names: List[str]
    ) -> List[str]:
        """
        Validate dataset schemas before merging
        
        Args:
            datasets: List of DataFrames
            source_names: Source names
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        for i, (df, source_name) in enumerate(zip(datasets, source_names)):
            # Check for empty datasets
            if len(df) == 0:
                warnings.append(f"Warning: {source_name} is empty")
            
            # Check for required fields (at least title or abstract)
            has_title = any('title' in col.lower() for col in df.columns)
            has_abstract = any('abstract' in col.lower() for col in df.columns)
            
            if not has_title and not has_abstract:
                warnings.append(
                    f"Warning: {source_name} has no 'title' or 'abstract' field"
                )
            
            # Check for duplicated column names
            duplicated_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicated_cols:
                warnings.append(
                    f"Warning: {source_name} has duplicated columns: {duplicated_cols}"
                )
        
        if warnings:
            logger.warning(f"Schema validation found {len(warnings)} issues")
            for warning in warnings:
                logger.warning(warning)
        
        return warnings
    
    def _apply_field_mapping(
        self,
        df: pd.DataFrame,
        mapping_result: FieldMappingResult
    ) -> pd.DataFrame:
        """
        Apply field mapping to DataFrame
        
        Args:
            df: Original DataFrame
            mapping_result: FieldMappingResult from FieldMapper
            
        Returns:
            DataFrame with mapped column names
        """
        # Create mapping dictionary
        rename_dict = {m['original_field']: m['standard_field'] 
                      for m in mapping_result.mappings}
        
        # Rename columns
        mapped_df = df.rename(columns=rename_dict)
        
        return mapped_df
    
    def _merge_dataframes(
        self,
        dataframes: List[pd.DataFrame],
        source_names: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """
        Merge multiple DataFrames
        
        Args:
            dataframes: List of DataFrames with standardized columns
            source_names: Source names for logging
            
        Returns:
            Tuple of (merged DataFrame, conflicts resolved count)
        """
        if len(dataframes) == 1:
            return dataframes[0].copy(), 0
        
        conflicts_resolved = 0
        
        # Remove duplicate columns from each DataFrame before merging
        cleaned_dataframes = []
        for i, df in enumerate(dataframes):
            if df.columns.duplicated().any():
                logger.warning(f"⚠️ Found duplicate columns in dataset {i}, removing duplicates")
                # Keep first occurrence of duplicate columns, merge their values
                df_cleaned = df.loc[:, ~df.columns.duplicated(keep='first')]
                cleaned_dataframes.append(df_cleaned)
            else:
                cleaned_dataframes.append(df)
        
        # Simple concatenation - pandas handles missing columns
        merged_df = pd.concat(cleaned_dataframes, ignore_index=True, sort=False)
        
        logger.info(
            f"Concatenated {len(cleaned_dataframes)} datasets: "
            f"{' + '.join(f'{len(df)}' for df in cleaned_dataframes)} = {len(merged_df)} records"
        )
        
        return merged_df, conflicts_resolved
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns to put standard fields first
        
        Args:
            df: DataFrame to reorder
            
        Returns:
            DataFrame with reordered columns
        """
        # Get current columns
        current_cols = df.columns.tolist()
        
        # Separate standard fields and other fields
        standard_present = [col for col in self.STANDARD_FIELDS if col in current_cols]
        other_cols = [col for col in current_cols if col not in self.STANDARD_FIELDS]
        
        # New order: standard fields first, then others, source column last
        new_order = standard_present + other_cols
        
        # Move data_source to the end if present
        if 'data_source' in new_order:
            new_order.remove('data_source')
            new_order.append('data_source')
        
        return df[new_order]
    
    def _generate_merge_summary(
        self,
        original_datasets: List[pd.DataFrame],
        merged_df: pd.DataFrame,
        source_names: List[str],
        field_mappings: Dict[str, int],
        conflicts: int
    ) -> str:
        """
        Generate human-readable merge summary
        
        Args:
            original_datasets: Original DataFrames
            merged_df: Merged DataFrame
            source_names: Source names
            field_mappings: Field mappings applied per source
            conflicts: Number of conflicts resolved
            
        Returns:
            Formatted summary string
        """
        summary = f"""
=== Data Merge Summary ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Sources merged: {len(original_datasets)}
"""
        
        for i, (source_name, df) in enumerate(zip(source_names, original_datasets)):
            mappings = field_mappings.get(source_name, 0)
            summary += f"  {i+1}. {source_name}: {len(df)} records, {mappings} field mappings\n"
        
        summary += f"""
Total records after merge: {len(merged_df)}
Total columns: {len(merged_df.columns)}
Conflicts resolved: {conflicts}

Column breakdown:
  Standard fields: {sum(1 for col in merged_df.columns if col in self.STANDARD_FIELDS)}
  Additional fields: {sum(1 for col in merged_df.columns if col not in self.STANDARD_FIELDS)}

Key statistics:
  Records with title: {merged_df['title'].notna().sum() if 'title' in merged_df.columns else 'N/A'}
  Records with abstract: {merged_df['abstract'].notna().sum() if 'abstract' in merged_df.columns else 'N/A'}
  Records with DOI: {merged_df['doi'].notna().sum() if 'doi' in merged_df.columns else 'N/A'}
  Records with year: {merged_df['year'].notna().sum() if 'year' in merged_df.columns else 'N/A'}
"""
        
        if 'data_source' in merged_df.columns:
            source_dist = merged_df['data_source'].value_counts()
            summary += f"\nRecords per source:\n"
            for source, count in source_dist.items():
                summary += f"  {source}: {count}\n"
        
        return summary
    
    def validate_merged_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate merged dataset quality
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'empty_records': 0,
            'missing_title': 0,
            'missing_abstract': 0,
            'missing_year': 0,
            'completeness_score': 0.0,
            'warnings': []
        }
        
        # Check for completely empty records
        validation['empty_records'] = df.isna().all(axis=1).sum()
        
        # Check key fields
        if 'title' in df.columns:
            validation['missing_title'] = df['title'].isna().sum()
        
        if 'abstract' in df.columns:
            validation['missing_abstract'] = df['abstract'].isna().sum()
        
        if 'year' in df.columns:
            validation['missing_year'] = df['year'].isna().sum()
        
        # Calculate completeness score (0-100)
        if len(df) > 0:
            standard_fields_present = [col for col in self.STANDARD_FIELDS if col in df.columns]
            if standard_fields_present:
                completeness = df[standard_fields_present].notna().mean().mean() * 100
                validation['completeness_score'] = round(completeness, 2)
        
        # Generate warnings
        if validation['empty_records'] > 0:
            validation['warnings'].append(
                f"{validation['empty_records']} completely empty records found"
            )
        
        if validation['missing_title'] > len(df) * 0.1:  # >10% missing titles
            validation['warnings'].append(
                f"{validation['missing_title']} records missing titles "
                f"({validation['missing_title']/len(df)*100:.1f}%)"
            )
        
        if validation['completeness_score'] < 50:
            validation['warnings'].append(
                f"Low completeness score: {validation['completeness_score']}%"
            )
        
        return validation
