import sys
sys.path.insert(0, '/home/nitai/code/platform/backend')

from websearch_clients.content_scraper import ContentScraper

# Test with a mix of working and failing URLs
test_urls = [
    "https://doi.org/10.3410/f.725503383.793537447",  # Works
    "https://doi.org/10.13182/fst10-8",  # Fails
    "https://pubmed.ncbi.nlm.nih.gov/30473232/",  # Works
]

scraper = ContentScraper(
    timeout=15,
    max_workers=3,
    max_content_length=10_000_000,
    pdf_extraction_mode="local",
    disable_agentic_fallback=True
)

print("Processing citations...")
citation_meta = scraper.process_citations(test_urls, fetch_content=True)
print(f"\nReturned {len(citation_meta)} metadata objects")

print("\n" + "=" * 80)
print("RAW METADATA:")
for i, meta in enumerate(citation_meta):
    print(f"\n{i+1}. URL: {meta.url}")
    print(f"   Content length: {len(meta.content) if meta.content else 0}")
    print(f"   Error: {meta.error}")
    print(f"   Has content: {bool(meta.content)}")

print("\n" + "=" * 80)
print("PREPARE_FOR_LLM OUTPUT:")
scraper_results = scraper.prepare_for_llm(citation_meta, include_metadata=True)
print(f"Returned {len(scraper_results)} results")
for url, content in scraper_results.items():
    print(f"\n  URL: {url}")
    print(f"  Content length: {len(content)}")
    print(f"  Starts with [ERROR]: {content.startswith('[ERROR]')}")
    if content.startswith('[ERROR]'):
        print(f"  Full content: {content}")
