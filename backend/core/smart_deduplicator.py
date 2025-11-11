"""
Smart Deduplicator - Intelligent Deduplication Engine

Deduplication Workflow:
1. Pre-processing - Remove invalid records (no title/abstract, short abstract)
2. DOI Exact Matching - Most reliable deduplication method
3. Title Similarity Matching - Identify potentially duplicate articles
4. Metadata Validation - Check author/journal/year for title-similar articles
   - Metadata matches → Confirmed duplicate
   - Metadata differs → Flag for manual review
5. Generate Manual Review List
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Data quality check result"""
    total_records: int
    invalid_records: int
    removed_no_title: int
    removed_no_abstract: int
    removed_short_abstract: int
    valid_records: int
    invalid_details: List[Dict] = field(default_factory=list)


@dataclass
class DeduplicationResult:
    """Deduplication result"""
    original_count: int
    invalid_count: int
    after_quality_check: int
    doi_duplicates: int
    title_duplicates_confirmed: int
    title_duplicates_need_review: int
    final_count: int
    
    # Detailed information
    strategies_used: List[str] = field(default_factory=list)
    quality_check: Optional[QualityCheckResult] = None
    
    # Records needing manual review
    needs_manual_review: List[Dict] = field(default_factory=list)
    
    # Cleaned data
    cleaned_dataframe: Optional[pd.DataFrame] = None
    review_dataframe: Optional[pd.DataFrame] = None  # Data needing manual review


class SmartDeduplicator:
    """
    Smart Deduplicator
    
    Features:
    1. Multi-stage quality check
    2. Intelligent metadata validation
    3. Manual review flagging
    4. Detailed deduplication report
    """
    
    def __init__(
        self,
        title_threshold: float = 0.85,
        min_abstract_length: int = 50,
        min_title_length: int = 10
    ):
        """
        Initialize Smart Deduplicator
        
        Args:
            title_threshold: Title similarity threshold (0.85 recommended)
            min_abstract_length: Minimum abstract length
            min_title_length: Minimum title length
        """
        self.title_threshold = title_threshold
        self.min_abstract_length = min_abstract_length
        self.min_title_length = min_title_length
    
    def deduplicate(self, df: pd.DataFrame) -> DeduplicationResult:
        """
        Execute complete deduplication workflow
        
        Args:
            df: DataFrame to deduplicate
            
        Returns:
            DeduplicationResult containing all deduplication information
        """
        logger.info(f"🚀 Starting smart deduplication workflow, total records: {len(df)}")
        original_count = len(df)
        
        # Add original index for tracking
        df = df.copy()
        df['_original_index'] = range(len(df))
        
        # ===== Step 1: Data Quality Check =====
        logger.info("📋 Step 1: Data Quality Check")
        df_valid, quality_result = self._quality_check(df)
        
        logger.info(
            f"✅ Quality check complete: {len(df_valid)}/{original_count} records valid "
            f"(removed {quality_result.invalid_records} invalid records)"
        )
        
        # ===== Step 2: DOI Exact Matching =====
        logger.info("📋 Step 2: DOI Exact Matching")
        df_after_doi, doi_duplicates = self._deduplicate_by_doi(df_valid)
        
        logger.info(f"✅ DOI deduplication complete: removed {doi_duplicates} duplicates")
        
        # ===== Step 3 & 4: Title Similarity + Metadata Validation =====
        logger.info("📋 Step 3-4: Title Similarity Matching + Metadata Validation")
        df_final, title_confirmed, needs_review = self._deduplicate_by_title_with_metadata(
            df_after_doi
        )
        
        logger.info(
            f"✅ Title deduplication complete: "
            f"confirmed duplicates {title_confirmed}, "
            f"needs manual review {len(needs_review)}"
        )
        
        # ===== Generate Result =====
        result = DeduplicationResult(
            original_count=original_count,
            invalid_count=quality_result.invalid_records,
            after_quality_check=len(df_valid),
            doi_duplicates=doi_duplicates,
            title_duplicates_confirmed=title_confirmed,
            title_duplicates_need_review=len(needs_review),
            final_count=len(df_final),
            quality_check=quality_result,
            needs_manual_review=needs_review,
            cleaned_dataframe=df_final.drop(columns=['_original_index'], errors='ignore'),
            review_dataframe=self._create_review_dataframe(needs_review) if needs_review else None,
            strategies_used=['quality_check', 'doi_matching', 'title_similarity', 'metadata_validation']
        )
        
        logger.info(
            f"🎉 Deduplication complete! {original_count} → {len(df_final)} "
            f"(removed {original_count - len(df_final)})"
        )
        
        return result
    
    def _quality_check(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityCheckResult]:
        """
        Step 1: Data Quality Check
        
        Removes:
        - Records without title
        - Records without abstract  
        - Records with too short abstract (< min_abstract_length characters)
        """
        total = len(df)
        invalid_details = []
        
        # Use standardized column names (after LLM mapping, should be lowercase)
        title_col = 'title' if 'title' in df.columns else None
        abstract_col = 'abstract' if 'abstract' in df.columns else None
        
        # If standardized columns don't exist, try to find alternatives
        if not title_col:
            title_col = self._find_column(df, ['Title', 'TI', 'Article Title'])
        if not abstract_col:
            abstract_col = self._find_column(df, ['Abstract', 'AB'])
        
        # Create validity mask
        is_valid = pd.Series([True] * len(df), index=df.index)
        
        removed_no_title = 0
        removed_no_abstract = 0
        removed_short_abstract = 0
        
        # Check title
        if title_col:
            no_title_mask = df[title_col].isna() | (df[title_col].astype(str).str.strip() == '')
            no_title_mask |= df[title_col].astype(str).str.len() < self.min_title_length
            
            for idx in df[no_title_mask].index:
                invalid_details.append({
                    'index': int(df.loc[idx, '_original_index']),
                    'reason': 'No title or title too short',
                    'title': str(df.loc[idx, title_col])[:50] if title_col in df.columns else 'N/A'
                })
            
            is_valid &= ~no_title_mask
            removed_no_title = no_title_mask.sum()
        
        # Check abstract
        if abstract_col:
            no_abstract_mask = df[abstract_col].isna() | (df[abstract_col].astype(str).str.strip() == '')
            short_abstract_mask = df[abstract_col].astype(str).str.len() < self.min_abstract_length
            
            for idx in df[no_abstract_mask].index:
                if is_valid[idx]:  # Only record if not already marked invalid
                    invalid_details.append({
                        'index': int(df.loc[idx, '_original_index']),
                        'reason': 'No abstract',
                        'title': str(df.loc[idx, title_col])[:50] if title_col else 'N/A'
                    })
            
            for idx in df[short_abstract_mask & ~no_abstract_mask].index:
                if is_valid[idx]:
                    abstract_len = len(str(df.loc[idx, abstract_col]))
                    invalid_details.append({
                        'index': int(df.loc[idx, '_original_index']),
                        'reason': f'Abstract too short ({abstract_len} chars)',
                        'title': str(df.loc[idx, title_col])[:50] if title_col else 'N/A',
                        'abstract': str(df.loc[idx, abstract_col])[:100]
                    })
            
            is_valid &= ~(no_abstract_mask | short_abstract_mask)
            removed_no_abstract = no_abstract_mask.sum()
            removed_short_abstract = (short_abstract_mask & ~no_abstract_mask).sum()
        
        # Filter valid records
        df_valid = df[is_valid].copy()
        
        quality_result = QualityCheckResult(
            total_records=total,
            invalid_records=(~is_valid).sum(),
            removed_no_title=removed_no_title,
            removed_no_abstract=removed_no_abstract,
            removed_short_abstract=removed_short_abstract,
            valid_records=len(df_valid),
            invalid_details=invalid_details[:100]  # Save max 100 details
        )
        
        return df_valid, quality_result
    
    def _deduplicate_by_doi(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Step 2: DOI-based exact deduplication
        
        Returns:
            (Deduplicated DataFrame, number of duplicates removed)
        """
        # Use standardized column name first
        doi_col = 'doi' if 'doi' in df.columns else None
        if not doi_col:
            doi_col = self._find_column(df, ['DOI', 'DI', 'AID'])
        
        if not doi_col:
            logger.warning("⚠️ DOI column not found, skipping DOI deduplication")
            return df, 0
        
        # Clean DOI (remove null and empty strings)
        df_with_doi = df[df[doi_col].notna() & (df[doi_col].astype(str).str.strip() != '')].copy()
        df_without_doi = df[df[doi_col].isna() | (df[doi_col].astype(str).str.strip() == '')].copy()
        
        if len(df_with_doi) == 0:
            return df, 0
        
        # Normalize DOI (lowercase, trim spaces)
        df_with_doi['_doi_clean'] = df_with_doi[doi_col].astype(str).str.lower().str.strip()
        
        # Keep first occurrence, remove subsequent duplicates
        before = len(df_with_doi)
        df_with_doi = df_with_doi.drop_duplicates(subset=['_doi_clean'], keep='first')
        after = len(df_with_doi)
        
        duplicates_removed = before - after
        
        # Remove temporary column
        df_with_doi = df_with_doi.drop(columns=['_doi_clean'])
        
        # Merge data with and without DOI
        df_result = pd.concat([df_with_doi, df_without_doi], ignore_index=True)
        
        return df_result, duplicates_removed
    
    def _deduplicate_by_title_with_metadata(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int, List[Dict]]:
        """
        Step 3-4: Title similarity matching + metadata validation
        
        Workflow:
        1. Find pairs of articles with similar titles
        2. For each similar pair:
           - Check if author/journal/year match
           - If match → Confirmed duplicate, remove
           - If don't match → Flag for manual review
        
        Returns:
            (Deduplicated DataFrame, confirmed duplicates count, list needing review)
        """
        # Use standardized column name first
        title_col = 'title' if 'title' in df.columns else None
        if not title_col:
            title_col = self._find_column(df, ['Title', 'TI'])
        
        if not title_col:
            logger.warning("⚠️ Title column not found, skipping title deduplication")
            return df, 0, []
        
        # Prepare title data
        titles = df[title_col].fillna('').astype(str)
        
        if len(titles) < 2:
            return df, 0, []
        
        # Calculate TF-IDF similarity
        logger.info("🔍 Calculating title similarity...")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(titles)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.error(f"❌ TF-IDF calculation failed: {e}")
            return df, 0, []
        
        # Find similar article pairs
        confirmed_duplicates = set()  # Confirmed duplicate indices
        needs_review = []  # Needs manual review
        
        for i in range(len(similarity_matrix)):
            if i in confirmed_duplicates:
                continue
            
            for j in range(i + 1, len(similarity_matrix)):
                if j in confirmed_duplicates:
                    continue
                
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.title_threshold:
                    # Title similar! Check metadata
                    metadata_match = self._check_metadata_match(df.iloc[i], df.iloc[j])
                    
                    if metadata_match:
                        # Metadata matches → Confirmed duplicate
                        confirmed_duplicates.add(j)  # Keep i, remove j
                        logger.debug(
                            f"✓ Confirmed duplicate (similarity={similarity:.3f}): "
                            f"'{titles.iloc[i][:50]}...' == '{titles.iloc[j][:50]}...'"
                        )
                    else:
                        # Metadata differs → Needs manual review
                        needs_review.append({
                            'index_1': int(df.iloc[i]['_original_index']),
                            'index_2': int(df.iloc[j]['_original_index']),
                            'title_1': str(titles.iloc[i]),
                            'title_2': str(titles.iloc[j]),
                            'similarity': float(similarity),
                            'reason': 'Title similar but metadata mismatch',
                            **self._extract_metadata_comparison(df.iloc[i], df.iloc[j])
                        })
                        logger.debug(
                            f"? Needs review (similarity={similarity:.3f}): "
                            f"'{titles.iloc[i][:50]}...' vs '{titles.iloc[j][:50]}...'"
                        )
        
        # Remove confirmed duplicates
        df_result = df.iloc[[i for i in range(len(df)) if i not in confirmed_duplicates]].copy()
        
        return df_result, len(confirmed_duplicates), needs_review
    
    def _check_metadata_match(self, row1: pd.Series, row2: pd.Series) -> bool:
        """
        Check if metadata of two records match
        
        Checks:
        - Authors (authors/Author/AU)
        - Journal (journal/Journal/SO/TA)
        - Year (year/Year/PY/DP)
        
        Returns:
            True if all available metadata matches
        """
        # Use standardized column names first
        author_col = 'authors' if 'authors' in row1.index else self._find_column_in_row(row1, ['Author', 'AU'])
        journal_col = 'journal' if 'journal' in row1.index else self._find_column_in_row(row1, ['Journal', 'SO', 'TA', 'Source title'])
        year_col = 'year' if 'year' in row1.index else self._find_column_in_row(row1, ['Year', 'PY', 'DP', 'Publication Year'])
        
        matches = []
        
        # Check authors
        if author_col:
            author1 = self._normalize_text(row1.get(author_col, ''))
            author2 = self._normalize_text(row2.get(author_col, ''))
            if author1 and author2:
                # Check if first author is the same
                first_author1 = author1.split(';')[0].split(',')[0].strip()
                first_author2 = author2.split(';')[0].split(',')[0].strip()
                matches.append(first_author1 == first_author2)
        
        # Check journal
        if journal_col:
            journal1 = self._normalize_text(row1.get(journal_col, ''))
            journal2 = self._normalize_text(row2.get(journal_col, ''))
            if journal1 and journal2:
                matches.append(journal1 == journal2)
        
        # Check year
        if year_col:
            year1 = self._extract_year(row1.get(year_col, ''))
            year2 = self._extract_year(row2.get(year_col, ''))
            if year1 and year2:
                matches.append(year1 == year2)
        
        # If no comparable metadata, return False (needs manual review)
        if not matches:
            return False
        
        # All available metadata must match
        return all(matches)
    
    def _extract_metadata_comparison(self, row1: pd.Series, row2: pd.Series) -> Dict:
        """Extract metadata for comparison report"""
        author_col = 'authors' if 'authors' in row1.index else self._find_column_in_row(row1, ['Author', 'AU'])
        journal_col = 'journal' if 'journal' in row1.index else self._find_column_in_row(row1, ['Journal', 'SO', 'TA'])
        year_col = 'year' if 'year' in row1.index else self._find_column_in_row(row1, ['Year', 'PY', 'DP'])
        abstract_col = 'abstract' if 'abstract' in row1.index else self._find_column_in_row(row1, ['Abstract', 'AB'])
        
        return {
            'author_1': str(row1.get(author_col, 'N/A'))[:100] if author_col else 'N/A',
            'author_2': str(row2.get(author_col, 'N/A'))[:100] if author_col else 'N/A',
            'journal_1': str(row1.get(journal_col, 'N/A'))[:100] if journal_col else 'N/A',
            'journal_2': str(row2.get(journal_col, 'N/A'))[:100] if journal_col else 'N/A',
            'year_1': str(row1.get(year_col, 'N/A')) if year_col else 'N/A',
            'year_2': str(row2.get(year_col, 'N/A')) if year_col else 'N/A',
            'abstract_1': str(row1.get(abstract_col, 'N/A'))[:200] if abstract_col else 'N/A',
            'abstract_2': str(row2.get(abstract_col, 'N/A'))[:200] if abstract_col else 'N/A'
        }
    
    def _create_review_dataframe(self, needs_review: List[Dict]) -> pd.DataFrame:
        """Create DataFrame for manual review"""
        if not needs_review:
            return pd.DataFrame()
        
        return pd.DataFrame(needs_review)
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column name in DataFrame"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _find_column_in_row(self, row: pd.Series, possible_names: List[str]) -> Optional[str]:
        """Find column name in Series"""
        for name in possible_names:
            if name in row.index:
                return name
        return None
    
    def _normalize_text(self, text: any) -> str:
        """Normalize text for comparison"""
        if pd.isna(text) or text == '':
            return ''
        return str(text).lower().strip()
    
    def _extract_year(self, text: any) -> Optional[str]:
        """Extract year from text"""
        if pd.isna(text):
            return None
        
        text_str = str(text)
        
        # Try to extract 4-digit year
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', text_str)
        if years:
            return years[0]
        
        return None
    
    def generate_report(self, result: DeduplicationResult) -> str:
        """Generate detailed deduplication report"""
        report = []
        report.append("=" * 80)
        report.append("Smart Deduplication Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        report.append("【Overall Statistics】")
        report.append(f"Original records: {result.original_count}")
        report.append(f"Final records: {result.final_count}")
        report.append(f"Records removed: {result.original_count - result.final_count}")
        report.append(f"Deduplication rate: {(result.original_count - result.final_count) / result.original_count * 100:.2f}%")
        report.append("")
        
        # Quality check
        if result.quality_check:
            qc = result.quality_check
            report.append("【Step 1: Data Quality Check】")
            report.append(f"Invalid records total: {qc.invalid_records}")
            report.append(f"  - No title or title too short: {qc.removed_no_title}")
            report.append(f"  - No abstract: {qc.removed_no_abstract}")
            report.append(f"  - Abstract too short (< {self.min_abstract_length} chars): {qc.removed_short_abstract}")
            report.append(f"Valid records: {qc.valid_records}")
            report.append("")
        
        # DOI deduplication
        report.append("【Step 2: DOI Exact Matching】")
        report.append(f"Duplicates removed: {result.doi_duplicates}")
        report.append("")
        
        # Title deduplication
        report.append("【Step 3-4: Title Similarity + Metadata Validation】")
        report.append(f"Title similarity threshold: {self.title_threshold}")
        report.append(f"Confirmed duplicates (metadata matched): {result.title_duplicates_confirmed}")
        report.append(f"Needs manual review (metadata mismatch): {result.title_duplicates_need_review}")
        report.append("")
        
        # Manual review list
        if result.needs_manual_review:
            report.append("【Records Needing Manual Review】")
            report.append(f"Total {len(result.needs_manual_review)} pairs need review")
            report.append("First 5 examples:")
            for i, item in enumerate(result.needs_manual_review[:5], 1):
                report.append(f"\n  {i}. Similarity: {item['similarity']:.3f}")
                report.append(f"     Title 1: {item['title_1'][:80]}...")
                report.append(f"     Title 2: {item['title_2'][:80]}...")
                report.append(f"     Author 1: {item.get('author_1', 'N/A')}")
                report.append(f"     Author 2: {item.get('author_2', 'N/A')}")
                report.append(f"     Year 1: {item.get('year_1', 'N/A')}, Year 2: {item.get('year_2', 'N/A')}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
