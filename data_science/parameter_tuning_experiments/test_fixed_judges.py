import sys
import os
sys.path.insert(0, '/home/nitai/code/platform/backend')
sys.path.insert(0, '/home/nitai/code/causalix.ai/backend')
sys.path.insert(0, '/home/nitai/code/causalix.ai/data_science')

from websearch_clients.content_scraper import ContentScraper
from lm_clients.openai_client_working import OpenAIClient
import logging

# Test with a URL that ContentScraper can fetch
test_url = "https://doi.org/10.3410/f.725503383.793537447"

print("Testing fixed judge with ContentScraper content...")
print("=" * 80)

# Fetch content with ContentScraper
scraper = ContentScraper(
    timeout=15,
    max_workers=1,
    max_content_length=10_000_000,
    pdf_extraction_mode="local",
    disable_agentic_fallback=True
)

citation_meta = scraper.process_citations([test_url], fetch_content=True)
scraper_results = scraper.prepare_for_llm(citation_meta, include_metadata=True)

if test_url in scraper_results:
    content = scraper_results[test_url]
    print(f"✅ ContentScraper fetched {len(content)} chars")
    print(f"Preview: {content[:200]}...")
    
    # Now test judge with this content
    print("\n" + "=" * 80)
    print("Testing OpenAI judge with enable_websearch=False...")
    
    judge = OpenAIClient(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        api_url=os.environ.get("OPENAI_API_URL", ""),
        model="gpt-4.1",
        enable_websearch=False,  # Web search disabled
        logger=logging.getLogger("test_judge")
    )
    
    claim = "ApoE-4 carriership directly causes an increase in Amyloid beta burden"
    sys_prompt = "You are a precise judge evaluating if a causal claim is supported by citation content."
    usr_prompt = f"CAUSAL CLAIM: {claim}\n\nCITATION CONTENT:\n{content[:2000]}\n\nRespond with: VERDICT: [SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED]\nREASON: [explanation]"
    
    try:
        response = judge.send_message(sys_prompt, usr_prompt, stream=False)
        data = judge.parse_static_response(response)
        llm_content = data.get("content", "")
        print(f"\n✅ Judge response received (no web search errors!):")
        print(f"{llm_content[:500]}...")
    except Exception as e:
        print(f"\n❌ Judge error: {e}")
else:
    print("❌ ContentScraper failed to fetch content")
