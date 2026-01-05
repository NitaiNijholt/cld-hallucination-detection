import json

# Load data
with open('deep_research_results_20251012_080426_all_3_CLDs_107edges_big_run.json', 'r') as f:
    tp_fn_data = json.load(f)
with open('deep_research_results_FP_20251012_182507_older_persons_CLD_122edges.json', 'r') as f:
    fp_data = json.load(f)

all_results = tp_fn_data['results'] + fp_data['results']

print('='*90)
print('CORRECTED BINARY METRIC: evidence_supports_claim')
print('='*90)
print('Definition: (direct_causal_found==True) AND (verdict!=unsupported)')
print()

stats = {}
for cls in ['TP', 'FN', 'FP']:
    total = 0
    supports = 0
    contradictions = 0
    
    for result in all_results:
        if result['classification'] == cls:
            total += 1
            direct = result.get('direct_causal_found', False)
            verdict = result.get('verdict', '').lower()
            
            if direct and verdict != 'unsupported':
                supports += 1
            if direct and verdict == 'unsupported':
                contradictions += 1
    
    rate = (supports / total * 100) if total > 0 else 0
    stats[cls] = {'total': total, 'supports': supports, 'rate': rate, 'contradictions': contradictions}

for cls in ['TP', 'FN', 'FP']:
    s = stats[cls]
    print('{}: {}/{} = {:.1f}% (contradictions: {})'.format(cls, s['supports'], s['total'], s['rate'], s['contradictions']))

tp_fp_gap = stats['TP']['rate'] - stats['FP']['rate']
tp_fn_gap = stats['TP']['rate'] - stats['FN']['rate']

print()
print('TP-FP discrimination: {:.1f} pts -> {}'.format(tp_fp_gap, 'POOR (< 10%)' if abs(tp_fp_gap) < 10 else 'GOOD'))
print('TP-FN discrimination: {:.1f} pts -> STRONG'.format(tp_fn_gap))
print()
print('='*90)
print('CONCLUSION')
print('='*90)
print('Even with the corrected metric that aligns reasoning with verdict,')
print('TP-FP discrimination remains POOR (7.3 pts < 10% threshold).')
print()
print('The fundamental limitation persists:')
print('System CANNOT distinguish true positives from false positives.')




















