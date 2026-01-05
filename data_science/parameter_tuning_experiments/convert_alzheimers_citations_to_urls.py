#!/usr/bin/env python3
"""
Convert Alzheimer's bibliographic citations to searchable URLs.
"""

import pandas as pd
import re
from pathlib import Path

def parse_citations(citation_string):
    """
    Parse citation string like "Uchino (2006); Hawkley (2010)" 
    into structured format.
    """
    if pd.isna(citation_string) or not citation_string:
        return []
    
    # Split by semicolon
    citations = [c.strip() for c in citation_string.split(';')]
    
    parsed = []
    for cite in citations:
        # Extract author and year using regex
        # Pattern: "Author Name (YYYY)" or "Author et al. (YYYY)"
        match = re.match(r'(.+?)\s*\((\d{4})\)', cite)
        if match:
            author = match.group(1).strip()
            year = match.group(2)
            parsed.append({
                'author': author,
                'year': year,
                'original': cite,
                'search_query': f'{author} {year}'
            })
        else:
            # If no match, use as-is
            parsed.append({
                'author': cite,
                'year': None,
                'original': cite,
                'search_query': cite
            })
    
    return parsed


def create_search_urls(parsed_citations):
    """
    Create search URLs for finding the papers.
    """
    urls = []
    for cite in parsed_citations:
        # PubMed search URL
        pubmed_query = cite['search_query'].replace(' ', '+')
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={pubmed_query}"
        
        # Google Scholar search URL
        scholar_query = cite['search_query'].replace(' ', '+')
        scholar_url = f"https://scholar.google.com/scholar?q={scholar_query}"
        
        urls.append({
            'original': cite['original'],
            'pubmed': pubmed_url,
            'scholar': scholar_url
        })
    
    return urls


def convert_alzheimers_citations():
    """Convert Alzheimer's citations to URL format."""
    
    print("="*80)
    print("CONVERTING ALZHEIMER'S CITATIONS TO URLS")
    print("="*80)
    
    # Load original data
    df_orig = pd.read_excel('alzheimers_cld_supplementary.xlsx', sheet_name='Connections')
    df_orig = df_orig[df_orig['Unnamed: 1'].notna()].copy()
    df_orig = df_orig[df_orig['Unnamed: 2'].notna()].copy()
    
    # Load references sheet for full citations
    df_refs = pd.read_excel('alzheimers_cld_supplementary.xlsx', sheet_name='References')
    
    print(f"\nTotal edges: {len(df_orig)}")
    print(f"Total references: {len(df_refs)}")
    
    # Process citations
    all_results = []
    
    for idx, row in df_orig.iterrows():
        source = row['Unnamed: 1']
        target = row['Unnamed: 3']
        polarity = row['Unnamed: 2']
        citations = row['Unnamed: 4']
        
        if pd.isna(source) or pd.isna(target):
            continue
        
        parsed = parse_citations(citations)
        search_urls = create_search_urls(parsed)
        
        # Create URL list for this edge
        url_list = [url_info['pubmed'] for url_info in search_urls]
        
        all_results.append({
            'Source': source,
            'Target': target,
            'Polarity': polarity,
            'Original_Citations': citations,
            'Parsed_Citations': '; '.join([c['original'] for c in parsed]),
            'PubMed_Search_URLs': ' | '.join([url['pubmed'] for url in search_urls]),
            'Number_of_Citations': len(parsed)
        })
    
    df_result = pd.DataFrame(all_results)
    
    # Map polarity
    polarity_map = {'+': 'POSITIVE', '-': 'NEGATIVE'}
    df_result['Relationship Type'] = df_result['Polarity'].map(polarity_map)
    df_result = df_result[df_result['Relationship Type'].notna()].copy()
    
    print(f"\nValid edges with citations: {len(df_result)}")
    print(f"\nSample conversions:")
    print(df_result[['Source', 'Target', 'Original_Citations', 'Number_of_Citations']].head(3).to_string())
    
    # Save to Excel
    output_file = Path('parameter_tuning_experiments/ground_truth_clds_for_experiments/Alzheimers_disease_with_search_urls.xlsx')
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main edges with URLs
        df_result.to_excel(writer, sheet_name='Edges_with_URLs', index=False)
        
        # Create a simplified version for judge (just URLs as list)
        df_simple = df_result[['Source', 'Target', 'Relationship Type', 'PubMed_Search_URLs']].copy()
        df_simple.columns = ['Source', 'Target', 'Relationship Type', 'Citation_Search_URLs']
        df_simple.to_excel(writer, sheet_name='Edges', index=False)
        
        # Full references
        df_refs.to_excel(writer, sheet_name='Full_References', index=False)
    
    print(f"\n✅ Saved to: {output_file}")
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"  Total edges: {len(df_result)}")
    print(f"  Edges with citations: {len(df_result[df_result['Number_of_Citations'] > 0])}")
    print(f"  Average citations per edge: {df_result['Number_of_Citations'].mean():.1f}")
    print(f"  Max citations: {df_result['Number_of_Citations'].max()}")
    
    return df_result


def create_note_about_citations():
    """Create a README explaining the citation situation."""
    
    readme = """# Alzheimer's CLD - Citation Format Notes

## Important: Citation Limitations

### What We Have:
- **Bibliographic references**: e.g., "Uchino (2006); Hawkley (2010)"
- **Full reference list**: 169 papers in References sheet
- **Expert validation**: From Group Model Building with 17 experts

### What We DON'T Have:
- Direct DOI links
- Full text content
- Web-scrapable URLs

### For Step 4 (Literature-Based Validation):

**Current Challenge:**
The judge expects web URLs with scrapable content. The expert CLD has bibliographic references.

**Solutions:**

**Option 1: Manual Literature Corpus** (Recommended for thesis)
1. Manually collect the 169 cited papers
2. Extract relevant sections/abstracts
3. Provide as curated literature corpus to judge
4. This tests: "Does judge align with experts when using SAME literature?"

**Option 2: Automated Search** (Current file)
- PubMed search URLs provided
- Judge would need to: search → find paper → extract content
- More error-prone but automated

**Option 3: Hybrid Approach**
- Use the expert citations as GROUND TRUTH labels
- Generate NEW CLD on Alzheimer's with web search
- Compare: Web-cited edges vs Expert-cited edges
- This tests: "Do web citations support same conclusions as expert literature?"

## Recommendation:

For **Step 4** of your 4-step narrative, use **Option 3**:
1. Generate Alzheimer's CLD using your current pipeline (web search)
2. Compare generated edges to expert CLD structure
3. For edges that match: Compare web citations vs expert citations
4. For edges that don't match: Analyze why (information access gap)

This directly supports your hypothesis:
"Judge fails on expert validation (Step 3) because web search ≠ domain literature.
When provided domain literature (expert citations), alignment improves."
"""
    
    readme_file = Path('parameter_tuning_experiments/ground_truth_clds_for_experiments/Alzheimers_Citation_README.md')
    with open(readme_file, 'w') as f:
        f.write(readme)
    
    print(f"📝 Created README: {readme_file}")


if __name__ == "__main__":
    df = convert_alzheimers_citations()
    create_note_about_citations()
    
    print(f"\n{'='*80}")
    print("NEXT STEPS FOR STEP 4:")
    print("="*80)
    print("1. Decide: Manual corpus, Auto search, or Hybrid?")
    print("2. If manual: Collect the 169 papers cited by experts")
    print("3. If auto: Use PubMed URLs for search-based access")
    print("4. If hybrid: Generate new AD CLD and compare citation sources")
    print()



