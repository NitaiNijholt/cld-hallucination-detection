"""
This file contains the parallelized version of judge_all_edges_with_citations_serial.
Apply this patch to modules.py to enable parallel judging.

The key change is replacing the sequential `for idx, record in enumerate(records, start=1):`
loop with either:
- Sequential processing using the helper method _process_single_edge_judgment
- Parallel processing using ThreadPoolExecutor when parallel=True
"""

# REPLACEMENT FOR THE LOOP STARTING AROUND LINE 3994 in judge_all_edges_with_citations_serial:
# Replace this section:
#     for idx, record in enumerate(records, start=1):
#         # ... all the edge processing logic ...
#
# With this:

"""
            # ------------------------------------------------------------
            # 4) Process edges - either sequentially or in parallel
            # ------------------------------------------------------------
            if parallel:
                # Parallel processing using ThreadPoolExecutor
                print(f"\\nProcessing {len(records)} edges in PARALLEL with {max_workers} workers...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all edge processing tasks
                    future_to_edge = {}
                    for idx, record in enumerate(records, start=1):
                        future = executor.submit(
                            self._process_single_edge_judgment,
                            idx=idx,
                            total_edges=len(records),
                            record=record,
                            ephemeral_judges=ephemeral_judges,
                            models=models,
                            approach=approach
                        )
                        future_to_edge[future] = (idx, record["source"], record["target"])
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_edge):
                        idx, src, tgt = future_to_edge[future]
                        try:
                            result = future.result()
                            if result:
                                judged_edges.append(result)
                            print(f"✓ Completed edge {idx}/{len(records)}: {src} -> {tgt}")
                        except Exception as e:
                            logger.error(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")
                            print(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")
            else:
                # Sequential processing
                print(f"\\nProcessing {len(records)} edges SEQUENTIALLY...")
                
                for idx, record in enumerate(records, start=1):
                    try:
                        result = self._process_single_edge_judgment(
                            idx=idx,
                            total_edges=len(records),
                            record=record,
                            ephemeral_judges=ephemeral_judges,
                            models=models,
                            approach=approach
                        )
                        if result:
                            judged_edges.append(result)
                    except Exception as e:
                        src = record.get("source", "?")
                        tgt = record.get("target", "?")
                        logger.error(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")
                        print(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")

            print(f"\\n{'='*50}")
            print(f"CITATION JUDGING (approach='{approach}'): Complete! Judged {len(judged_edges)} edges.")
            print(f"{'='*50}\\n")

            # --------------------------------------------------------------------
            # 5) Merge ephemeral usage stats into self.judge_llm for final reporting
            # --------------------------------------------------------------------
            for ephemeral_llm in ephemeral_judges:
                ephemeral_stats = ephemeral_llm.get_stats()
                self._merge_judge_usage(ephemeral_stats)

            return judged_edges

        except Exception as e:
            logger.error(f"CITATION JUDGING ERROR (outer loop): {e}")
            print(f"CITATION JUDGING ERROR (outer loop): {e}")
            return []
"""

# INSTRUCTIONS:
# 1. Find line ~3994 in modules.py where the loop starts: `for idx, record in enumerate(records, start=1):`
# 2. Delete the entire loop (which goes for about 600+ lines until the final return statement)
# 3. Replace with the code block above (without the triple quotes)
# 4. Make sure the indentation matches (4 levels of indentation for the main if/else block)
