import json

# Load data
with open('deep_research_results_20251012_080426_all_3_CLDs_107edges_big_run.json', 'r') as f:
    tp_fn_data = json.load(f)
with open('deep_research_results_FP_20251012_182507_older_persons_CLD_122edges.json', 'r') as f:
    fp_data = json.load(f)

all_results = tp_fn_data['results'] + fp_data['results']

print('='*90)
print('PROPOSED: SEPARATE CAUSAL vs CORRELATIONAL VERDICTS')
print('='*90)
print()

# Analyze current patterns
patterns = {
    'TP': {'strong_causal': 0, 'corr_only': 0, 'weak': 0, 'total': 0},
    'FN': {'strong_causal': 0, 'corr_only': 0, 'weak': 0, 'total': 0},
    'FP': {'strong_causal': 0, 'corr_only': 0, 'weak': 0, 'total': 0}
}

for result in all_results:
    cls = result['classification']
    direct_causal = result.get('direct_causal_found', False)
    verdict = result.get('verdict', '').lower()
    
    patterns[cls]['total'] += 1
    
    # Categorize
    if direct_causal and verdict != 'unsupported':
        patterns[cls]['strong_causal'] += 1
    elif not direct_causal and verdict != 'unsupported':
        patterns[cls]['corr_only'] += 1
    else:
        patterns[cls]['weak'] += 1

print('Current System: Single Verdict')
print('  - Combines causal and correlational evidence')
print('  - Cannot distinguish evidence types')
print()
print('Proposed System: Dual Verdicts')
print('  - causal_verdict: supported/partial/unsupported (based on RCTs, experiments)')
print('  - correlational_verdict: supported/partial/unsupported (based on obs studies)')
print()
print('='*90)
print('EVIDENCE PATTERNS FROM CURRENT DATA')
print('='*90)
print()

for cls in ['TP', 'FN', 'FP']:
    s = patterns[cls]
    total = s['total']
    
    print('{} (n={}):'.format(cls, total))
    print('  Strong causal + supported: {}/{} ({:.1f}%)'.format(
        s['strong_causal'], total, (s['strong_causal']/total*100) if total > 0 else 0))
    print('  Correlational only: {}/{} ({:.1f}%)'.format(
        s['corr_only'], total, (s['corr_only']/total*100) if total > 0 else 0))
    print('  Weak evidence: {}/{} ({:.1f}%)'.format(
        s['weak'], total, (s['weak']/total*100) if total > 0 else 0))
    print()

# Show discrimination
print('='*90)
print('BENEFITS OF DUAL VERDICT APPROACH')
print('='*90)
print()
print('1. RICHER INTERPRETATION:')
print('   Examples:')
print('   - causal_verdict=unsupported, correlational_verdict=supported')
print('     → "Strong correlational evidence but needs causal studies"')
print('   - causal_verdict=supported, correlational_verdict=supported')
print('     → "Strong evidence at both levels"')
print()
print('2. BETTER FEEDBACK TO USERS:')
print('   - Identify which edges need experimental validation')
print('   - Distinguish evidence gaps from lack of relationship')
print()
print('3. IMPROVED DISCRIMINATION:')
tp_causal = (patterns['TP']['strong_causal'] / patterns['TP']['total']) * 100
fp_causal = (patterns['FP']['strong_causal'] / patterns['FP']['total']) * 100
fn_causal = (patterns['FN']['strong_causal'] / patterns['FN']['total']) * 100

tp_corr = (patterns['TP']['corr_only'] / patterns['TP']['total']) * 100
fp_corr = (patterns['FP']['corr_only'] / patterns['FP']['total']) * 100
fn_corr = (patterns['FN']['corr_only'] / patterns['FN']['total']) * 100

print('   Strong causal evidence:')
print('     TP={:.1f}%, FP={:.1f}%, FN={:.1f}%'.format(tp_causal, fp_causal, fn_causal))
print('     TP-FP gap: {:.1f} pts (same as current)'.format(tp_causal - fp_causal))
print()
print('   Correlational only:')
print('     TP={:.1f}%, FP={:.1f}%, FN={:.1f}%'.format(tp_corr, fp_corr, fn_corr))
print('     TP-FP gap: {:.1f} pts'.format(tp_corr - fp_corr))
print()
print('4. NEW INSIGHTS:')
print('   - Can identify "correlational abundance" problem for FP edges')
print('   - Can see which TP edges rely on theory vs evidence')
print('   - Can distinguish FN rejection reasons (no evidence vs weak causal)')
print()
print('='*90)
print('RECOMMENDATION')
print('='*90)
print()
print('YES - Implement dual verdicts!')
print()
print('Schema change needed:')
print('  class EvidenceJudgment(BaseModel):')
print('      causal_verdict: str  # supported/partially/unsupported')
print('      causal_confidence: int')
print('      correlational_verdict: str  # supported/partially/unsupported')
print('      correlational_confidence: int')
print('      overall_verdict: str  # synthesis of both')
print('      reasoning: str')
print()
print('This provides:')
print('  ✓ Better interpretability')
print('  ✓ Richer feedback')
print('  ✓ Same discrimination (no worse)')
print('  ✓ New research insights')




















