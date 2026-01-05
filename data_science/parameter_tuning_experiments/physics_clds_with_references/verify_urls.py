#!/usr/bin/env python3
"""
Verify and display statistics about URLs in physics CLD reference files.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_cld_file(filepath: Path) -> dict:
    """Analyze URLs in a CLD file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'filename': filepath.name,
        'cld_name': data.get('cld_name', 'Unknown'),
        'total_refs': 0,
        'refs_with_urls': 0,
        'url_types': Counter(),
        'sample_refs': []
    }
    
    # Process primary references
    if "primary_references" in data:
        for ref in data["primary_references"]:
            stats['total_refs'] += 1
            if ref.get('url'):
                stats['refs_with_urls'] += 1
                url = ref['url']
                if 'doi.org' in url:
                    stats['url_types']['DOI'] += 1
                elif 'archive.org' in url:
                    stats['url_types']['Archive.org'] += 1
                elif '.pdf' in url:
                    stats['url_types']['Open Access PDF'] += 1
                elif any(pub in url for pub in ['wiley.com', 'pearson.com', 'mheducation.com', 'oup.com', 'cengage.com']):
                    stats['url_types']['Publisher Page'] += 1
                elif 'google.com/books' in url:
                    stats['url_types']['Google Books'] += 1
                elif 'wikipedia.org' in url:
                    stats['url_types']['Wikipedia'] += 1
                else:
                    stats['url_types']['Other'] += 1
                
                # Collect samples
                if len(stats['sample_refs']) < 3:
                    stats['sample_refs'].append({
                        'citation': ref['short_citation'],
                        'url': url
                    })
    
    # Process edge references
    if "edges" in data:
        for edge in data["edges"]:
            if "references" in edge:
                for ref in edge["references"]:
                    stats['total_refs'] += 1
                    if ref.get('url'):
                        stats['refs_with_urls'] += 1
                        url = ref['url']
                        if 'doi.org' in url:
                            stats['url_types']['DOI'] += 1
                        elif 'archive.org' in url:
                            stats['url_types']['Archive.org'] += 1
                        elif '.pdf' in url:
                            stats['url_types']['Open Access PDF'] += 1
                        elif any(pub in url for pub in ['wiley.com', 'pearson.com', 'mheducation.com', 'oup.com', 'cengage.com']):
                            stats['url_types']['Publisher Page'] += 1
                        elif 'google.com/books' in url:
                            stats['url_types']['Google Books'] += 1
                        elif 'wikipedia.org' in url:
                            stats['url_types']['Wikipedia'] += 1
                        else:
                            stats['url_types']['Other'] += 1
    
    return stats

def main():
    """Analyze all CLD files and display statistics."""
    cld_dir = Path(__file__).parent
    json_files = sorted(cld_dir.glob("*_with_refs.json"))
    
    print("=" * 80)
    print("PHYSICS CLD REFERENCE URL VERIFICATION")
    print("=" * 80)
    
    all_stats = []
    total_refs = 0
    total_with_urls = 0
    overall_url_types = Counter()
    
    for json_file in json_files:
        stats = analyze_cld_file(json_file)
        all_stats.append(stats)
        total_refs += stats['total_refs']
        total_with_urls += stats['refs_with_urls']
        overall_url_types.update(stats['url_types'])
    
    # Display per-CLD statistics
    print(f"\n{'CLD':<30} {'Total Refs':<12} {'With URLs':<12} {'Coverage':<10}")
    print("-" * 80)
    for stats in all_stats:
        coverage = (stats['refs_with_urls'] / stats['total_refs'] * 100) if stats['total_refs'] > 0 else 0
        print(f"{stats['cld_name']:<30} {stats['total_refs']:<12} {stats['refs_with_urls']:<12} {coverage:>6.1f}%")
    
    # Display overall statistics
    print("-" * 80)
    overall_coverage = (total_with_urls / total_refs * 100) if total_refs > 0 else 0
    print(f"{'TOTAL':<30} {total_refs:<12} {total_with_urls:<12} {overall_coverage:>6.1f}%")
    
    # Display URL type breakdown
    print(f"\n{'URL Type':<30} {'Count':<10} {'Percentage'}")
    print("-" * 80)
    for url_type, count in overall_url_types.most_common():
        percentage = (count / total_with_urls * 100) if total_with_urls > 0 else 0
        print(f"{url_type:<30} {count:<10} {percentage:>6.1f}%")
    
    # Display sample URLs for each CLD
    print("\n" + "=" * 80)
    print("SAMPLE URLs FROM EACH CLD")
    print("=" * 80)
    for stats in all_stats:
        print(f"\n{stats['cld_name']}:")
        for i, sample in enumerate(stats['sample_refs'], 1):
            print(f"  {i}. {sample['citation']}")
            print(f"     → {sample['url']}")
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - {len(all_stats)} physics CLDs processed")
    print(f"  - {total_refs} total references")
    print(f"  - {total_with_urls} references with URLs ({overall_coverage:.1f}% coverage)")
    print(f"  - {len(overall_url_types)} different URL types")
    print(f"\nURL accessibility:")
    print(f"  - Open access (Archive.org, PDFs, Wikipedia): {overall_url_types['Archive.org'] + overall_url_types['Open Access PDF'] + overall_url_types['Wikipedia']} references")
    print(f"  - DOI links: {overall_url_types['DOI']} references")
    print(f"  - Publisher pages (may require purchase): {overall_url_types['Publisher Page']} references")

if __name__ == "__main__":
    main()
