# Ablation Study: Flag-Set Comparison

Each configuration disables a subset of the detector's flags via constructor `thresholds=` overrides. The detector code (src/detector.py) is not modified.


| Configuration | FP rate | impossible recall | pingpong recall | skipped_step recall | unlikely_branch recall |
|---|---|---|---|---|---|
| full_5_flag | 4.62% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_PT_surprise | 4.62% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_PT_rank | 4.62% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_PT_surprise_and_rank | 4.62% | 100.0% | 100.0% | 100.0% | 100.0% |
| minimal_PT_imp_plus_seq | 4.62% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_SQ_low_likelihood | 1.74% | 100.0% | 100.0% | 100.0% | 0.0% |
| no_SQ_edge_repeat | 3.58% | 100.0% | 0.0% | 100.0% | 100.0% |


## Interpretation

If `minimal_PT_imp_plus_seq` matches `full_5_flag` on recall, the redundancy hypothesis is empirically confirmed and PT-rank / PT-surprise can be demoted to evidence-pack metadata.
