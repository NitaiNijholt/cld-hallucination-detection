# Alzheimer's CLD - Citation Format Notes

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
