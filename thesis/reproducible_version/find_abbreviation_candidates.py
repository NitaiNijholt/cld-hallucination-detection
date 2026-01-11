#!/usr/bin/env python3
"""
N-gram frequency analysis for LaTeX thesis files.
Finds frequently used multi-word phrases that could be abbreviated.
"""

import re
import os
from collections import Counter
from pathlib import Path

try:
    import nltk
    from nltk import ngrams
    from nltk.tokenize import word_tokenize
except ImportError:
    print("Installing nltk...")
    import subprocess
    subprocess.check_call(["pip", "install", "nltk"])
    import nltk
    from nltk import ngrams
    from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def clean_latex(text: str) -> str:
    """Remove LaTeX commands and keep readable text."""
    # Remove comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    
    # Remove common LaTeX environments we don't want
    text = re.sub(r'\\begin\{(equation|align|figure|table|lstlisting|tikzpicture)\*?\}.*?\\end\{\1\*?\}', '', text, flags=re.DOTALL)
    
    # Keep text inside \textbf{}, \textit{}, \emph{}, etc.
    text = re.sub(r'\\(textbf|textit|texttt|emph|underline)\{([^}]*)\}', r'\2', text)
    
    # Remove \cite{}, \ref{}, \label{}, etc.
    text = re.sub(r'\\(cite[pt]?|ref|label|eqref|autoref|hyperref)\{[^}]*\}', '', text)
    
    # Remove \section{}, \subsection{}, etc. but keep the title
    text = re.sub(r'\\(section|subsection|subsubsection|paragraph|chapter)\*?\{([^}]*)\}', r'\2', text)
    
    # Remove remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', '', text)
    
    # Remove special characters and brackets
    text = re.sub(r'[{}\[\]$\\~^&]', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_ngrams(text: str, n: int) -> list:
    """Extract n-grams from text."""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Filter out non-alphabetic tokens and very short words
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
    
    # Generate n-grams
    return list(ngrams(tokens, n))


def is_meaningful_ngram(ngram: tuple, stopwords: set) -> bool:
    """Check if an n-gram is meaningful (not just stopwords)."""
    # At least one word should not be a stopword
    meaningful_words = [w for w in ngram if w not in stopwords]
    return len(meaningful_words) >= len(ngram) // 2 + 1


def analyze_thesis(directory: str, existing_abbreviations: list = None):
    """Analyze all .tex files in directory for n-gram frequencies."""
    
    # Common stopwords to filter
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'we', 'our', 'they', 'their',
        'which', 'who', 'whom', 'where', 'when', 'why', 'how', 'what',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'then', 'thus'
    }
    
    # Existing abbreviations (already abbreviated terms to exclude)
    if existing_abbreviations is None:
        existing_abbreviations = [
            'causal loop diagram', 'large language model', 'uncertainty quantification',
            'retrieval augmented generation', 'deep research', 'ground truth',
            'false positive', 'false negative', 'true positive', 'chain of thought',
            'computational science lab', 'universiteit van amsterdam'
        ]
    
    # Collect all text
    all_text = ""
    tex_files = list(Path(directory).rglob("*.tex"))
    
    print(f"Found {len(tex_files)} .tex files")
    
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                all_text += " " + clean_latex(content)
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
    
    print(f"Total text length: {len(all_text)} characters")
    print()
    
    # Analyze different n-gram sizes
    results = {}
    
    for n in [2, 3, 4]:
        print(f"{'='*60}")
        print(f"  {n}-GRAM ANALYSIS (phrases of {n} words)")
        print(f"{'='*60}")
        
        ngram_list = get_ngrams(all_text, n)
        
        # Filter meaningful n-grams
        meaningful = [ng for ng in ngram_list if is_meaningful_ngram(ng, stopwords)]
        
        # Count frequencies
        counter = Counter(meaningful)
        
        # Get top results
        top_ngrams = counter.most_common(50)
        
        print(f"\nTop {n}-word phrases (count >= 5):\n")
        print(f"{'Count':<8} {'Phrase':<40} {'Possible Abbrev'}")
        print("-" * 70)
        
        for ngram, count in top_ngrams:
            if count < 5:
                break
            phrase = ' '.join(ngram)
            
            # Skip if already abbreviated
            if phrase in existing_abbreviations:
                continue
            
            # Generate possible abbreviation
            abbrev = ''.join(w[0].upper() for w in ngram)
            
            # Mark if particularly good candidate
            marker = ""
            if count >= 20:
                marker = " ⭐ HIGH PRIORITY"
            elif count >= 10:
                marker = " ✓ Good candidate"
            
            print(f"{count:<8} {phrase:<40} {abbrev}{marker}")
            
            results[(n, phrase)] = count
        
        print()
    
    # Summary of best candidates
    print(f"\n{'='*60}")
    print("  RECOMMENDED ABBREVIATIONS (count >= 10)")
    print(f"{'='*60}\n")
    
    print("Add these to your LaTeX abbreviations list:\n")
    print("\\listofsymbols{ll}")
    print("{")
    
    all_candidates = [(count, n, phrase) for (n, phrase), count in results.items() if count >= 10]
    all_candidates.sort(reverse=True)
    
    for count, n, phrase in all_candidates[:20]:
        abbrev = ''.join(w[0].upper() for w in phrase.split())
        phrase_formatted = phrase.title()
        abbrev_expanded = ' '.join(f"\\textbf{{{w[0].upper()}}}{w[1:]}" for w in phrase.split())
        print(f"\\textbf{{{abbrev}}} & {abbrev_expanded}\\\\ % ({count} occurrences)")
    
    print("}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default to current directory or take from command line
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    print(f"\nAnalyzing LaTeX files in: {os.path.abspath(directory)}\n")
    analyze_thesis(directory)










