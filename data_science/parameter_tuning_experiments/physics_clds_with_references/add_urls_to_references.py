#!/usr/bin/env python3
"""
Script to add URLs to all references in physics CLD JSON files.
URLs are constructed from DOIs (preferred) or ISBNs (for books).
"""

import json
from pathlib import Path
from typing import Dict, List

# Mapping of references to their URLs
# Priority: DOI > Open access link > Publisher page > Google Books
URL_MAPPING = {
    # Newton
    "Newton (1701)": "https://doi.org/10.1098/rstl.1701.0017",
    
    # Sterman - widely available
    "Sterman (2000)": "https://www.mheducation.com/highered/product/business-dynamics-systems-thinking-modeling-complex-world-sterman/M9780072389159.html",
    
    # Åström & Murray - Open access!
    "Åström & Murray (2008)": "https://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-complete_30Aug11.pdf",
    
    # Franklin et al.
    "Franklin et al. (2015)": "https://www.pearson.com/en-us/subject-catalog/p/feedback-control-of-dynamic-systems/P200000003484",
    
    # Incropera et al.
    "Incropera et al. (2007)": "https://www.wiley.com/en-us/Fundamentals+of+Heat+and+Mass+Transfer%2C+6th+Edition-p-9780471457282",
    
    # Çengel & Boles
    "Çengel & Boles (2015)": "https://www.mheducation.com/highered/product/thermodynamics-engineering-approach-cengel-boles/M9780073398174.html",
    
    # Holman
    "Holman (2010)": "https://www.mheducation.com/highered/product/heat-transfer-holman/M9780073529363.html",
    
    # ASHRAE
    "ASHRAE (2017)": "https://www.ashrae.org/technical-resources/ashrae-handbook",
    
    # Kreith et al.
    "Kreith et al. (2011)": "https://www.cengage.com/c/principles-of-heat-transfer-7e-kreith/9780495667704",
    
    # Lotka
    "Lotka (1925)": "https://archive.org/details/elementsofphysic017171mbp",
    
    # Volterra
    "Volterra (1926)": "https://www.nature.com/articles/118558a0",
    
    # Murray - has DOI
    "Murray (2002)": "https://doi.org/10.1007/b98868",
    
    # Gotelli
    "Gotelli (2008)": "https://global.oup.com/ushe/product/a-primer-of-ecology-9780878933181",
    
    # Begon et al.
    "Begon et al. (2006)": "https://www.wiley.com/en-us/Ecology%3A+From+Individuals+to+Ecosystems%2C+4th+Edition-p-9781405111171",
    
    # Kirchhoff
    "Kirchhoff (1845)": "https://doi.org/10.1002/andp.18451400402",
    
    # Ohm - Historical work
    "Ohm (1827)": "https://archive.org/details/diegalvanischek01ohmgoog",
    
    # Nilsson & Riedel
    "Nilsson & Riedel (2015)": "https://www.pearson.com/en-us/subject-catalog/p/electric-circuits/P200000003182",
    
    # Alexander & Sadiku
    "Alexander & Sadiku (2017)": "https://www.mheducation.com/highered/product/fundamentals-electric-circuits-alexander-sadiku/M9780078028229.html",
    
    # Halliday et al.
    "Halliday et al. (2013)": "https://www.wiley.com/en-us/Fundamentals+of+Physics%2C+10th+Edition-p-9781118230718",
    
    # Maxwell
    "Maxwell (1873)": "https://archive.org/details/electricandmagne01maxwrich",
    
    # Purcell & Morin - has DOI
    "Purcell & Morin (2013)": "https://doi.org/10.1017/CBO9781139012973",
    
    # Faraday
    "Faraday (1839)": "https://archive.org/details/experimentalrese01fara",
    
    # Torricelli - Historical work
    "Torricelli (1643)": "https://en.wikipedia.org/wiki/Evangelista_Torricelli#Opera_geometrica",
    
    # Pascal - Historical work
    "Pascal (1663)": "https://archive.org/details/traitdelquili00pasc",
    
    # White
    "White (2016)": "https://www.mheducation.com/highered/product/fluid-mechanics-white/M9780073398273.html",
    
    # Munson et al.
    "Munson et al. (2013)": "https://www.wiley.com/en-us/Fundamentals+of+Fluid+Mechanics%2C+7th+Edition-p-9781118116135",
    
    # Bernoulli - Historical work
    "Bernoulli (1738)": "https://archive.org/details/hydrodynamica00bern",
}

def add_url_to_reference(ref: Dict) -> Dict:
    """Add URL field to a reference object based on short_citation."""
    short_cite = ref.get("short_citation", "")
    
    # If already has URL, skip
    if "url" in ref:
        return ref
    
    # Try to get URL from mapping
    if short_cite in URL_MAPPING:
        ref["url"] = URL_MAPPING[short_cite]
    # If has DOI, construct DOI URL
    elif "doi" in ref:
        ref["url"] = f"https://doi.org/{ref['doi']}"
    # If has ISBN, construct Google Books URL
    elif "isbn" in ref:
        isbn = ref["isbn"].replace("-", "")
        ref["url"] = f"https://www.google.com/books?isbn={isbn}"
    else:
        # No URL available
        ref["url"] = None
        print(f"  WARNING: No URL found for {short_cite}")
    
    return ref

def process_cld_file(filepath: Path) -> None:
    """Process a single CLD JSON file and add URLs to all references."""
    print(f"\nProcessing: {filepath.name}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add URLs to primary references
    if "primary_references" in data:
        for ref in data["primary_references"]:
            add_url_to_reference(ref)
    
    # Add URLs to edge references
    if "edges" in data:
        for edge in data["edges"]:
            if "references" in edge:
                for ref in edge["references"]:
                    add_url_to_reference(ref)
    
    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Updated {filepath.name}")

def main():
    """Process all CLD JSON files in the directory."""
    cld_dir = Path(__file__).parent
    json_files = list(cld_dir.glob("*_with_refs.json"))
    
    print(f"Found {len(json_files)} CLD files to process")
    
    for json_file in sorted(json_files):
        process_cld_file(json_file)
    
    print(f"\n✅ All files updated with URLs!")
    print(f"\nURL sources:")
    print(f"  - DOI links: {sum(1 for url in URL_MAPPING.values() if 'doi.org' in url)}")
    print(f"  - Archive.org: {sum(1 for url in URL_MAPPING.values() if 'archive.org' in url)}")
    print(f"  - Publisher pages: {sum(1 for url in URL_MAPPING.values() if any(pub in url for pub in ['wiley.com', 'pearson.com', 'mheducation.com', 'oup.com', 'cengage.com']))}")
    print(f"  - Open access PDFs: {sum(1 for url in URL_MAPPING.values() if '.pdf' in url)}")

if __name__ == "__main__":
    main()
