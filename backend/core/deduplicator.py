"""
Deduplicator - Intelligent Duplicate Detection Engine

Multi-strategy duplicate detection preserving the excellent logic from modern_app.py
while improving structure, performance, and maintainability.

Strategies:
1. DOI exact matching (highest priority)
2. TF-IDF title similarity with cosine distance ≥ 0.85
3. Author-Year-Journal metadata combination

Refined improvements:
- Batch processing for large datasets
- Progress callbacks for UI updates
- Detailed duplicate reports with pairs
- Configurable thresholds
- Better error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation"""
    original_count: int
    duplicate_count: int
    final_count: int
    strategies_used: List[str] = field(default_factory=list)
    doi_duplicates: int = 0
    title_duplicates: int = 0
    metadata_duplicates: int = 0
    duplicate_pairs: List[Dict] = field(default_factory=list)  # Sample of duplicate pairs with details
    cleaned_dataframe: Optional[pd.DataFrame] = None


class Deduplicator:
    """
    Intelligent duplicate detector for systematic review data
    
    Uses multiple strategies to identify duplicates:
    - DOI matching (most reliable)
    - Title similarity using TF-IDF (cosine ≥ 0.85)
    - Author-Year-Journal combinations
    """
    
    def __init__(
        self,
        title_threshold: float = 0.85,
        min_title_length: int = 30,
        use_doi: bool = True,
        use_title_similarity: bool = True,
        use_metadata: bool = True
    ):
        """
        Initialize deduplicator
        
        Args:
            title_threshold: Cosine similarity threshold for title matching (0-1)
            min_title_length: Minimum title length for comparison
            use_doi: Enable DOI-based matching
            use_title_similarity: Enable TF-IDF title similarity matching
            use_metadata: Enable author-year-journal matching
        """
        self.title_threshold = title_threshold
        self.min_title_length = min_title_length
        self.use_doi = use_doi
        self.use_title_similarity = use_title_similarity
        self.use_metadata = use_metadata
    
    def find_duplicates(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> DeduplicationResult:
        """
        Find all duplicates using enabled strategies
        
        Args:
            df: DataFrame to check for duplicates
            progress_callback: Optional callback function(step, total_steps, message)
            
        Returns:
            DeduplicationResult with details and cleaned DataFrame
        """
        logger.info(f"Starting deduplication on {len(df)} records")
        original_count = len(df)
        
        # Track all duplicate indices
        all_duplicates = set()
        strategies_used = []
        
        # Track duplicates by strategy
        doi_dups = []
        title_dups = []
        metadata_dups = []
        duplicate_pairs = []
        
        total_steps = sum([self.use_doi, self.use_title_similarity, self.use_metadata])
        current_step = 0
        
        # Strategy 1: DOI matching
        if self.use_doi:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "Finding DOI duplicates...")
            
            doi_dups = self._find_doi_duplicates(df)
            if doi_dups:
                all_duplicates.update(doi_dups)
                strategies_used.append(f"DOI matching ({len(doi_dups)} found)")
                # Sample duplicate pairs
                duplicate_pairs.extend(self._get_duplicate_pairs(df, doi_dups[:5], 'DOI'))
                logger.info(f"Found {len(doi_dups)} DOI duplicates")
        
        # Strategy 2: Title similarity
        if self.use_title_similarity and 'title' in df.columns:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "Analyzing title similarity...")
            
            title_dups = self._find_title_duplicates(df)
            if title_dups:
                new_duplicates = set(title_dups) - all_duplicates
                all_duplicates.update(new_duplicates)
                strategies_used.append(f"Title similarity ({len(new_duplicates)} unique found)")
                # Sample duplicate pairs
                duplicate_pairs.extend(self._get_duplicate_pairs(df, list(new_duplicates)[:5], 'Title'))
                logger.info(f"Found {len(new_duplicates)} new title duplicates")
        
        # Strategy 3: Metadata combination
        if self.use_metadata:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "Checking metadata combinations...")
            
            metadata_dups = self._find_metadata_duplicates(df)
            if metadata_dups:
                new_duplicates = set(metadata_dups) - all_duplicates
                all_duplicates.update(new_duplicates)
                strategies_used.append(f"Metadata matching ({len(new_duplicates)} unique found)")
                # Sample duplicate pairs
                duplicate_pairs.extend(self._get_duplicate_pairs(df, list(new_duplicates)[:5], 'Metadata'))
                logger.info(f"Found {len(new_duplicates)} new metadata duplicates")
        
        # Remove duplicates
        duplicate_indices = list(all_duplicates)
        cleaned_df = df.drop(duplicate_indices).reset_index(drop=True)
        
        result = DeduplicationResult(
            original_count=original_count,
            duplicate_count=len(duplicate_indices),
            final_count=len(cleaned_df),
            strategies_used=strategies_used,
            doi_duplicates=len(doi_dups),
            title_duplicates=len(title_dups),
            metadata_duplicates=len(metadata_dups),
            duplicate_pairs=duplicate_pairs,
            cleaned_dataframe=cleaned_df
        )
        
        logger.info(
            f"Deduplication complete: {original_count} → {len(cleaned_df)} "
            f"({len(duplicate_indices)} duplicates removed)"
        )
        
        return result
    
    def _find_doi_duplicates(self, df: pd.DataFrame) -> List[int]:
        """
        Find duplicates based on DOI exact matching
        
        Args:
            df: DataFrame with 'doi' column
            
        Returns:
            List of indices to remove (keeps first occurrence)
        """
        if 'doi' not in df.columns:
            logger.warning("No 'doi' column found, skipping DOI matching")
            return []
        
        # Clean DOI field
        doi_cleaned = df['doi'].fillna('').astype(str).str.strip().str.lower()
        
        # Only check non-empty DOIs
        non_empty_mask = doi_cleaned != ''
        
        if not non_empty_mask.any():
            logger.info("No non-empty DOIs found")
            return []
        
        # Find duplicates (keeping first occurrence)
        duplicates_mask = df[non_empty_mask].duplicated(subset=['doi'], keep='first')
        duplicate_indices = df[non_empty_mask][duplicates_mask].index.tolist()
        
        return duplicate_indices
    
    def _find_title_duplicates(self, df: pd.DataFrame) -> List[int]:
        """
        Find duplicates using TF-IDF title similarity
        
        Uses cosine similarity with threshold of 0.85 (or configured)
        
        Args:
            df: DataFrame with 'title' column
            
        Returns:
            List of indices to remove
        """
        if 'title' not in df.columns:
            logger.warning("No 'title' column found, skipping title matching")
            return []
        
        try:
            # Clean titles
            titles_clean = df['title'].apply(self._clean_title)
            
            # Filter titles that are long enough
            valid_mask = titles_clean.str.len() >= self.min_title_length
            
            if valid_mask.sum() <= 1:
                logger.info("Not enough valid titles for similarity comparison")
                return []
            
            valid_titles = titles_clean[valid_mask]
            valid_indices = df.index[valid_mask].tolist()
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                dtype=np.float32  # Memory optimization
            )
            
            tfidf_matrix = vectorizer.fit_transform(valid_titles)
            
            # Calculate cosine similarity
            # For large datasets, process in batches to avoid memory issues
            if len(valid_titles) > 1000:
                duplicates = self._find_similar_pairs_batched(
                    tfidf_matrix, valid_indices, self.title_threshold
                )
            else:
                cosine_sim = cosine_similarity(tfidf_matrix)
                duplicates = self._find_similar_pairs(cosine_sim, valid_indices)
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Title similarity matching failed: {e}")
            return []
    
    def _find_metadata_duplicates(self, df: pd.DataFrame) -> List[int]:
        """
        Find duplicates based on Author-Year-Journal combination
        
        Args:
            df: DataFrame with 'authors', 'year', 'journal' columns
            
        Returns:
            List of indices to remove
        """
        required_cols = ['authors', 'year', 'journal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for metadata matching: {missing_cols}")
            # Create missing columns with empty values
            for col in missing_cols:
                df[col] = ''
        
        # Create combined key
        df['_temp_combined_key'] = (
            df['authors'].astype(str).str[:50].str.lower() + "_" +
            df['year'].astype(str) + "_" +
            df['journal'].astype(str).str[:30].str.lower()
        )
        
        # Find duplicates
        duplicates_mask = df.duplicated(subset=['_temp_combined_key'], keep='first')
        duplicate_indices = df[duplicates_mask].index.tolist()
        
        # Clean up temporary column
        df.drop('_temp_combined_key', axis=1, inplace=True)
        
        return duplicate_indices
    
    @staticmethod
    def _clean_title(text: str) -> str:
        """
        Clean title for comparison
        
        - Lowercase
        - Remove punctuation
        - Normalize whitespace
        
        Args:
            text: Title string
            
        Returns:
            Cleaned title
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove punctuation, keep only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _find_similar_pairs(
        self,
        cosine_sim: np.ndarray,
        indices: List[int]
    ) -> List[int]:
        """
        Find similar pairs from cosine similarity matrix
        
        Args:
            cosine_sim: Cosine similarity matrix
            indices: Original DataFrame indices
            
        Returns:
            List of indices to remove (keeps first of each pair)
        """
        duplicates = []
        
        for i in range(len(cosine_sim)):
            for j in range(i + 1, len(cosine_sim)):
                if cosine_sim[i][j] >= self.title_threshold:
                    # Keep first (i), remove second (j)
                    duplicates.append(indices[j])
        
        return list(set(duplicates))  # Remove any double-counts
    
    def _find_similar_pairs_batched(
        self,
        tfidf_matrix,
        indices: List[int],
        threshold: float,
        batch_size: int = 500
    ) -> List[int]:
        """
        Find similar pairs using batched processing for large datasets
        
        Args:
            tfidf_matrix: TF-IDF sparse matrix
            indices: Original DataFrame indices
            threshold: Similarity threshold
            batch_size: Size of batches for processing
            
        Returns:
            List of indices to remove
        """
        duplicates = set()
        n = tfidf_matrix.shape[0]
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_sim = cosine_similarity(
                tfidf_matrix[i:end_i],
                tfidf_matrix
            )
            
            # Find duplicates in this batch
            for local_i in range(batch_sim.shape[0]):
                global_i = i + local_i
                for j in range(global_i + 1, n):
                    if batch_sim[local_i, j] >= threshold:
                        duplicates.add(indices[j])
        
        return list(duplicates)
    
    def _get_duplicate_pairs(
        self,
        df: pd.DataFrame,
        duplicate_indices: List[int],
        strategy: str
    ) -> List[Dict]:
        """
        Get sample of duplicate pairs for reporting
        
        Args:
            df: Original DataFrame
            duplicate_indices: Indices of duplicates found
            strategy: Which strategy found these duplicates
            
        Returns:
            List of duplicate pair dictionaries
        """
        pairs = []
        
        for idx in duplicate_indices[:5]:  # Limit to 5 samples
            if idx >= len(df):
                continue
            
            record = df.iloc[idx]
            pair_info = {
                'strategy': strategy,
                'duplicate_index': int(idx),
                'title': record.get('title', '')[:100] if 'title' in df.columns else '',
                'doi': record.get('doi', '') if 'doi' in df.columns else '',
                'year': record.get('year', '') if 'year' in df.columns else ''
            }
            pairs.append(pair_info)
        
        return pairs
    
    def generate_report(self, result: DeduplicationResult) -> str:
        """
        Generate human-readable deduplication report
        
        Args:
            result: DeduplicationResult object
            
        Returns:
            Formatted report string
        """
        report = f"""
=== Deduplication Report ===

Original records: {result.original_count}
Duplicates found: {result.duplicate_count}
Final unique records: {result.final_count}
Removal rate: {result.duplicate_count / result.original_count * 100:.1f}%

Strategies used:
{chr(10).join(f'  - {s}' for s in result.strategies_used)}

Breakdown by strategy:
  - DOI duplicates: {result.doi_duplicates}
  - Title duplicates: {result.title_duplicates}
  - Metadata duplicates: {result.metadata_duplicates}

Sample duplicate pairs ({len(result.duplicate_pairs)} shown):
"""
        
        for i, pair in enumerate(result.duplicate_pairs, 1):
            report += f"\n{i}. {pair['strategy']} match:\n"
            report += f"   Title: {pair['title']}\n"
            if pair['doi']:
                report += f"   DOI: {pair['doi']}\n"
            if pair['year']:
                report += f"   Year: {pair['year']}\n"
        
        return report
