import sys
sys.path.insert(0, '/home/nitai/code/platform/backend')

from websearch_clients.content_scraper import ContentScraper

# Test problematic URLs from the log
test_urls = [
    "https://doi.org/10.3410/f.725503383.793537447",  # OpenAlex URL
    "https://doi.org/10.13182/fst10-8",  # Common failure
    "https://doi.org/10.1161/hyp.0000000000000053",  # Another failure
    "https://pubmed.ncbi.nlm.nih.gov/30473232/",  # PubMed URL from Jina run
]

print("Testing ContentScraper with problematic URLs...")
print("=" * 80)

scraper = ContentScraper(
    timeout=15,
    max_workers=1,
    max_content_length=10_000_000,
    pdf_extraction_mode="local",
    disable_agentic_fallback=True
)

for url in test_urls:
    print(f"\nTesting: {url}")
    print("-" * 80)
    
    citation_meta = scraper.process_citations([url], fetch_content=True)
    scraper_results = scraper.prepare_for_llm(citation_meta, include_metadata=True)
    
    if url in scraper_results:
        content = scraper_results[url]
        print(f"✅ Content length: {len(content)}")
        print(f"Preview: {content[:300]}...")
    else:
        print(f"❌ No content returned")
        print(f"Metadata: {citation_meta}")
        if citation_meta:
            meta = citation_meta[0]
            print(f"  - URL: {meta.url}")
            print(f"  - Domain: {meta.domain}")
            print(f"  - Is valid: {meta.is_valid}")
            print(f"  - Status code: {meta.status_code}")
            print(f"  - Error: {meta.error}")
            print(f"  - Content: {meta.content[:200] if meta.content else 'None'}")

print("\n" + "=" * 80)
print("Test complete")
