#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Aggregator - Precise text segment aggregation based on sentence position markers

Main functions:
1. Detect overlaps between text segments based on 【Begin-x】【End-y】 markers
2. Automatically merge and reconstruct based on original order when overlapping
3. Retain the highest scoring segments
"""

import re
from typing import List, Tuple

class TextAggregator:
    """
    Text aggregator for merging retrieved text segments
    Implements splitting, deduplication, sorting, and reconstruction of text segments based on 【Begin-x】【End-x】 markers
    """
    
    def __init__(self):
        """
        Initialize text aggregator
        """
        pass
    
    def _extract_segments_from_text(self, text: str) -> List[Tuple[int, str]]:
        """
        Extract all 【Begin-x】...【End-x】 segments from text
        
        Args:
            text: Text containing position markers
            
        Returns:
            List[Tuple[int, str]]: List of (begin_index, segment_text)
        """
        segments = []
        # Match 【Begin-x】...【End-x】 pattern
        pattern = r'【Begin-(\d+)】(.*?)【End-\1】'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            begin_idx = int(match[0])
            segment_content = match[1]
            full_segment = f"【Begin-{begin_idx}】{segment_content}【End-{begin_idx}】"
            segments.append((begin_idx, full_segment))
        
        return segments
    
    def _remove_boundary_markers(self, text: str) -> str:
        """
        Remove all boundary markers from text, keeping only content
        
        Args:
            text: Text containing boundary markers
            
        Returns:
            str: Text with boundary markers removed
        """
        # Remove 【Begin-x】 and 【End-x】 markers
        clean_text = re.sub(r'【Begin-\d+】|【End-\d+】', '', text)
        return clean_text.strip()
    

    
    def aggregate_segments(self, segments: List[str]) -> str:
        """
        Aggregate text segments: split, deduplicate, sort, reconstruct
        
        Args:
            segments: List of text segments
            
        Returns:
            str: Aggregated text string
        """
        if not segments:
            return ""
        
        # Step 1: Extract segments from all input texts
        all_segments = {}  # {begin_index: segment_text}
        
        for text in segments:
            extracted = self._extract_segments_from_text(text)
            for begin_idx, segment in extracted:
                # Deduplication: Keep only one segment for the same begin_index
                if begin_idx not in all_segments:
                    all_segments[begin_idx] = segment
        
        # Step 2: Sort by begin_index
        sorted_segments = sorted(all_segments.items())
        
        # Step 3: Reconstruct text
        if not sorted_segments:
            return []
        
        # Build continuous text
        result_text = ""
        prev_end = -1
        
        for begin_idx, segment in sorted_segments:
            # If not continuous, add ellipsis
            if prev_end != -1 and begin_idx != prev_end + 1:
                result_text += "..."
            
            # Add content of current segment (remove boundary markers)
            content = self._remove_boundary_markers(segment)
            result_text += content
            
            prev_end = begin_idx
        
        return result_text
    
    def aggregate_segments_complete(self, segments: List[str]) -> str:
        """
        Completely aggregate all text segments
        
        Args:
            segments: List of text segments
            
        Returns:
            str: Aggregated text string
        """
        return self.aggregate_segments(segments)
    



def demo():
    """Demo function - Show text splitting, deduplication, sorting, and reconstruction based on position markers"""
    print("=== Text Aggregator Demo (Completely Rewritten Version) ===\n")
    
    # Create aggregator
    aggregator = TextAggregator()
    
    # Test data - Format according to user example
    test_segments = [
        "【Begin-1】sdfsdf【End-1】【Begin-2】sdfsdf【End-2】",
        "【Begin-2】sdfsdf【End-2】【Begin-3】sdfsdf【End-3】",
        "【Begin-5】sdfsdf【End-5】【Begin-6】sdfsdf【End-6】"
    ]
    
    print("Original input segments:")
    for i, text in enumerate(test_segments, 1):
        print(f"{i}. {text}")
    
    print("\n=== Step 1: Extract segments from each text ===")
    all_extracted = {}
    for i, text in enumerate(test_segments, 1):
        extracted = aggregator._extract_segments_from_text(text)
        print(f"Segments extracted from text {i}: {extracted}")
        for begin_idx, segment in extracted:
            if begin_idx not in all_extracted:
                all_extracted[begin_idx] = segment
                print(f"  Add segment: Begin-{begin_idx}")
            else:
                print(f"  Skip duplicate segment: Begin-{begin_idx}")
    
    print(f"\nAll segments after deduplication: {list(all_extracted.keys())}")
    
    print("\n=== Step 2: Sort by Begin marker ===")
    sorted_segments = sorted(all_extracted.items())
    print("Sorted segments:")
    for begin_idx, segment in sorted_segments:
        print(f"  Begin-{begin_idx}: {segment}")
    
    print("\n=== Step 3: Reconstruct text (remove boundary markers, add ellipsis) ===")
    result = aggregator.aggregate_segments(test_segments)
    print(f"Final result: {result}")
    
    print("\n=== Full Test Cases ===")
    
    # More complex test cases
    complex_segments = [
        "【Begin-1】First sentence【End-1】【Begin-2】Second sentence【End-2】【Begin-3】Third sentence【End-3】",
        "【Begin-2】Second sentence【End-2】【Begin-3】Third sentence【End-3】【Begin-4】Fourth sentence【End-4】",
        "【Begin-6】Sixth sentence【End-6】【Begin-7】Seventh sentence【End-7】",
        "【Begin-4】Fourth sentence【End-4】【Begin-5】Fifth sentence【End-5】"
    ]
    
    print("\nComplex test input:")
    for i, text in enumerate(complex_segments, 1):
        print(f"{i}. {text}")
    
    complex_result = aggregator.aggregate_segments(complex_segments)
    print(f"\nComplex test result: {complex_result}")
    
    print("\n=== Boundary Case Tests ===")
    
    # Test empty input
    empty_result = aggregator.aggregate_segments([])
    print(f"Empty input result: {empty_result}")
    
    # Test single segment
    single_result = aggregator.aggregate_segments(["【Begin-1】Single segment【End-1】"])
    print(f"Single segment result: {single_result}")
    
    # Test text without markers (should return empty)
    no_marker_result = aggregator.aggregate_segments(["Normal text without markers"])
    print(f"Text without markers result: {no_marker_result}")
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    demo()