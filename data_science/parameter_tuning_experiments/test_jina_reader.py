#!/usr/bin/env python3
"""
Test Jina AI Reader vs ContentScraper for citation extraction.

This script tests both methods on a sample of DOI URLs to compare success rates.
"""

import sys
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import ContentScraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample DOI URLs from the failed judging results
TEST_URLS = [
    "https://doi.org/10.1016/j.smrv.2017.06.010",
    "https://doi.org/10.1016/j.pathol.2018.11.002",
    "https://doi.org/10.13182/fst10-8",
    "https://doi.org/10.1111/dom.12524",
    "https://doi.org/10.1136/bmjopen-2015-008222",
    "https://pubmed.ncbi.nlm.nih.gov/27885006/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10000289/",
]

def test_content_scraper(url):
    """Test the current ContentScraper method."""
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing ContentScraper on: {url}")
        logger.info(f"{'='*80}")
        
        scraper = ContentScraper(
            timeout=15,
            max_workers=1,
            max_content_length=10_000_000,
            pdf_extraction_mode="local",
            disable_agentic_fallback=True
        )
        
        # Process single citation
        citation_meta = scraper.process_citations([url], fetch_content=True)
        citation_text_map = scraper.prepare_for_llm(citation_meta, include_metadata=True)
        
        if url in citation_text_map and citation_text_map[url]:
            content = citation_text_map[url]
            content_length = len(content)
            logger.info(f"✅ SUCCESS - Extracted {content_length} characters")
            logger.info(f"Preview: {content[:500]}...")
            return True, content_length
        else:
            logger.warning(f"❌ FAILED - No content extracted")
            return False, 0
            
    except Exception as e:
        logger.error(f"❌ ERROR - {str(e)}")
        return False, 0

def test_jina_reader(url):
    """Test Jina AI Reader method."""
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Jina AI Reader on: {url}")
        logger.info(f"{'='*80}")
        
        # Use Jina AI Reader
        jina_url = f"https://r.jina.ai/{url}"
        logger.info(f"Jina URL: {jina_url}")
        
        response = requests.get(jina_url, timeout=30)
        response.raise_for_status()
        
        content = response.text
        content_length = len(content)
        
        if content and content_length > 100:  # At least some meaningful content
            logger.info(f"✅ SUCCESS - Extracted {content_length} characters")
            logger.info(f"Preview: {content[:500]}...")
            return True, content_length
        else:
            logger.warning(f"❌ FAILED - Content too short ({content_length} chars)")
            return False, 0
            
    except Exception as e:
        logger.error(f"❌ ERROR - {str(e)}")
        return False, 0

def main():
    print("\n" + "="*80)
    print("JINA AI READER vs CONTENT SCRAPER TEST")
    print("="*80)
    print(f"Testing {len(TEST_URLS)} sample URLs\n")
    
    results = {
        "scraper": {"success": 0, "failed": 0, "total_chars": 0},
        "jina": {"success": 0, "failed": 0, "total_chars": 0}
    }
    
    for i, url in enumerate(TEST_URLS, 1):
        print(f"\n{'#'*80}")
        print(f"TEST {i}/{len(TEST_URLS)}: {url}")
        print(f"{'#'*80}")
        
        # Test ContentScraper
        scraper_success, scraper_chars = test_content_scraper(url)
        if scraper_success:
            results["scraper"]["success"] += 1
            results["scraper"]["total_chars"] += scraper_chars
        else:
            results["scraper"]["failed"] += 1
        
        # Test Jina AI Reader
        jina_success, jina_chars = test_jina_reader(url)
        if jina_success:
            results["jina"]["success"] += 1
            results["jina"]["total_chars"] += jina_chars
        else:
            results["jina"]["failed"] += 1
        
        print(f"\nResult for {url}:")
        print(f"  ContentScraper: {'✅ SUCCESS' if scraper_success else '❌ FAILED'} ({scraper_chars:,} chars)")
        print(f"  Jina AI Reader: {'✅ SUCCESS' if jina_success else '❌ FAILED'} ({jina_chars:,} chars)")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nContentScraper:")
    print(f"  ✅ Success: {results['scraper']['success']}/{len(TEST_URLS)} ({results['scraper']['success']/len(TEST_URLS)*100:.1f}%)")
    print(f"  ❌ Failed: {results['scraper']['failed']}/{len(TEST_URLS)}")
    print(f"  📄 Total chars: {results['scraper']['total_chars']:,}")
    print(f"  📊 Avg chars/success: {results['scraper']['total_chars']/max(1, results['scraper']['success']):,.0f}")
    
    print(f"\nJina AI Reader:")
    print(f"  ✅ Success: {results['jina']['success']}/{len(TEST_URLS)} ({results['jina']['success']/len(TEST_URLS)*100:.1f}%)")
    print(f"  ❌ Failed: {results['jina']['failed']}/{len(TEST_URLS)}")
    print(f"  📄 Total chars: {results['jina']['total_chars']:,}")
    print(f"  📊 Avg chars/success: {results['jina']['total_chars']/max(1, results['jina']['success']):,.0f}")
    
    print(f"\n🏆 Winner:")
    if results['jina']['success'] > results['scraper']['success']:
        improvement = (results['jina']['success'] - results['scraper']['success']) / max(1, results['scraper']['success']) * 100
        print(f"   Jina AI Reader ({results['jina']['success']} vs {results['scraper']['success']} successes)")
        if results['scraper']['success'] > 0:
            print(f"   Improvement: +{improvement:.0f}%")
        else:
            print(f"   Improvement: Infinite (from 0 to {results['jina']['success']})")
    elif results['scraper']['success'] > results['jina']['success']:
        print(f"   ContentScraper ({results['scraper']['success']} vs {results['jina']['success']} successes)")
    else:
        print(f"   Tie ({results['jina']['success']} successes each)")
    
    print("="*80)
    
    # Recommendation
    if results['jina']['success'] > results['scraper']['success']:
        print("\n✅ RECOMMENDATION: Use Jina AI Reader for judging")
    elif results['jina']['success'] == 0 and results['scraper']['success'] == 0:
        print("\n⚠️ WARNING: Both methods failed completely. Need alternative approach.")
    else:
        print("\n🤔 INCONCLUSIVE: Review results before deciding")

if __name__ == "__main__":
    main()



