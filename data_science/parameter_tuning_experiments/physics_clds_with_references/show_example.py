#!/usr/bin/env python3
import json

with open('thermostat_heating_system_with_refs.json', 'r') as f:
    data = json.load(f)

# Show one complete edge as example
edge = data['edges'][4]  # Room Temperature -> Heat Loss Rate

print("=" * 80)
print("EXAMPLE: Complete Edge with References and Supporting Quotes")
print("=" * 80)
print(f"\n📍 EDGE: {edge['source']} → {edge['target']}")
print(f"   Polarity: {edge['polarity']}")
print(f"\n💡 Motivation:")
print(f"   {edge['motivation']}")
print(f"\n📚 REFERENCES ({len(edge['references'])} total):\n")

for i, ref in enumerate(edge['references'], 1):
    print(f"   [{i}] {ref['short_citation']}")
    print(f"       📖 Citation: {ref['full_citation']}")
    print(f"       🔗 URL: {ref['url']}")
    if 'doi' in ref:
        print(f"       🔬 DOI: {ref['doi']}")
    print(f"       📄 Pages: {ref['pages']}")
    print(f"       🎯 Relevance: {ref['relevance']}")
    print(f"       💬 Supporting Quote:")
    print(f"          \"{ref['supporting_quote']}\"")
    print()

print("=" * 80)
