# Prompt archive (thesis deliverable)

This directory stores **copies** of all prompts used in the thesis experiments, organized **by Research Question (RQ)** to minimize thesis PDF clutter while preserving reproducibility.

## Structure
- `PRELIM_generator/`: generator prompt selection + ablations (edge prompt variants + final Nitai_C config + node prompt variants)
- `RQ1_verification_truthfulqa/`: TruthfulQA verification prompt(s)
- `RQ1a_judge/`: correctness + citation judge prompt variants
- `RQ1b_corrector/`: corrector prompt variants (in `corrector_prompts.yaml`)
- `RQ2_uq_metrics/`: no new prompts; see generator/judge prompts above
- `RQ3_deep_research/`: Deep Research prompt components (extracted system prompts)
- `shared/`: prompts reused across RQs (if any)

## Canonical IDs (aligned to thesis terminology)
- `RQ1a.judge_correctness.{baseline,cot,mechanistic}`
- `RQ1a.judge_citation.{baseline,cot,mechanistic,mech-lit}`
- `RQ1b.corrector.{baseline,cot,mechanistic}` (variants are inside `RQ1b_corrector/corrector/corrector_prompts.yaml`)
- `PRELIM.generator.edge_prompt.{Current,Nitai_A,Nitai_B,Nitai_C,Nitai_E,Rick_A,Rick_B,Cillian_B,Cillian_C}`
- `PRELIM.generator.node_prompt.{Node_V0_Minimal,Node_V1_Role_Only,Node_V2_Constraints,Node_V3_CoT_System,Node_V4_CoT_User,Node_V5_Nitai_C,Node_V6_Andrew}`
- `PRELIM.generator.final.Nitai_C` (contains `prompt_id: generateNodes` + `prompt_id: generateEdges`)

## File inventory (sha256 prefix)

| Path | Bytes | SHA256 (prefix) |
|---|---:|---|
| `PRELIM_generator/edge_ablation/Cillian_B.yaml` | 15322 | `7cde8535fb69` |
| `PRELIM_generator/edge_ablation/Cillian_C.yaml` | 15434 | `8e31f61f79bc` |
| `PRELIM_generator/edge_ablation/Current.yaml` | 16863 | `5d5a84cfdac7` |
| `PRELIM_generator/edge_ablation/Nitai_A.yaml` | 15029 | `1501246debd3` |
| `PRELIM_generator/edge_ablation/Nitai_B.yaml` | 17221 | `fb7bbdbec394` |
| `PRELIM_generator/edge_ablation/Nitai_C.yaml` | 20897 | `4037d77f5fc8` |
| `PRELIM_generator/edge_ablation/Nitai_E.yaml` | 17984 | `64f58b81da3f` |
| `PRELIM_generator/edge_ablation/Rick_A.yaml` | 16765 | `b82555b754ad` |
| `PRELIM_generator/edge_ablation/Rick_B.yaml` | 16953 | `4a532dc1d347` |
| `PRELIM_generator/final_generator/prompts_Nitai_C.yaml` | 17245 | `700a7fbf4d5e` |
| `PRELIM_generator/node_ablation/Node_V0_Minimal.yaml` | 1657 | `f90f28f47406` |
| `PRELIM_generator/node_ablation/Node_V1_Role_Only.yaml` | 1882 | `a22106355dc0` |
| `PRELIM_generator/node_ablation/Node_V2_Constraints.yaml` | 2045 | `31ecc7963b4a` |
| `PRELIM_generator/node_ablation/Node_V3_CoT_System.yaml` | 2227 | `6c333845f226` |
| `PRELIM_generator/node_ablation/Node_V4_CoT_User.yaml` | 2523 | `098b909c411c` |
| `PRELIM_generator/node_ablation/Node_V5_Nitai_C.yaml` | 3896 | `eb1c8fd36507` |
| `PRELIM_generator/node_ablation/Node_V6_Andrew.yaml` | 3710 | `cf65588d649a` |
| `README.md` | 3174 | `f291225cba97` |
| `RQ1_verification_truthfulqa/judge_correctness/prompts_truthqa_baseline.yaml` | 1625 | `bdb780b07e6a` |
| `RQ1a_judge/judge_citation/prompts_citation_baseline.yaml` | 20834 | `6408be9d5606` |
| `RQ1a_judge/judge_citation/prompts_citation_cot.yaml` | 20919 | `04c5b3083b44` |
| `RQ1a_judge/judge_citation/prompts_citation_mechanistic.yaml` | 23666 | `f7e541931617` |
| `RQ1a_judge/judge_citation/prompts_citation_mechanistic_lit.yaml` | 23844 | `a0c5dcb524a4` |
| `RQ1a_judge/judge_correctness/prompts_correctness_baseline.yaml` | 38455 | `0e7132b710f9` |
| `RQ1a_judge/judge_correctness/prompts_correctness_cot.yaml` | 20727 | `e6e7e9ebd003` |
| `RQ1a_judge/judge_correctness/prompts_correctness_mechanistic.yaml` | 24369 | `a21d06dc2401` |
| `RQ1b_corrector/corrector/corrector_prompts.yaml` | 21235 | `57908eeac81f` |
| `RQ3_deep_research/deep_research/pydantic_ai_deepresearch_multidimensional.py.extracted_prompts.txt` | 25644 | `8f42c35f77fd` |
| `RQ3_deep_research/deep_research/pydantic_ai_deepresearch_orchestration.py.extracted_prompts.txt` | 7829 | `0c65aee30695` |
