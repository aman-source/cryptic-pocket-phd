# Task A1 Stratified Protein Selection Report

## Summary
- Total eligible mdCATH domains: 5294
- Excluded (Lewis 33 + unknown class + out-of-range): 104
- Final selected: 100
- Selection seed: 42

## CATH Class Distribution

| CATH Class | Name | Count |
|---|---|---|
| 1 | Mainly_Alpha | 32 |
| 2 | Mainly_Beta | 32 |
| 3 | Alpha_Beta | 28 |
| 4 | Few_SS | 8 |

## Length Distribution

| Bin | Range | Count |
|---|---|---|
| 0 | 50-149 res | 30 |
| 1 | 150-249 res | 24 |
| 2 | 250-349 res | 23 |
| 3 | 350-499 res | 23 |

## Cell Breakdown (CATH Class × Length Bin)

| CATH Class | 50-149 | 150-249 | 250-349 | 350-499 | Total |
|---|---|---|---|---|---|
| Mainly_Alpha | 8 | 8 | 8 | 8 | 32 |
| Mainly_Beta | 8 | 8 | 8 | 8 | 32 |
| Alpha_Beta | 7 | 7 | 7 | 7 | 28 |
| Few_SS | 7 | 1 | 0 | 0 | 8 |

## Length Statistics
- Min: 55 residues
- Max: 472 residues
- Median: 216 residues

## GPU Split
- pod1_gpu0.txt: proteins 0,4,8,... (interleaved)
- pod1_gpu1.txt: proteins 1,5,9,... (interleaved)
- pod2_gpu0.txt: proteins 2,6,10,... (interleaved)
- pod2_gpu1.txt: proteins 3,7,11,... (interleaved)

## MMseqs2 Dedup Note
This list is pre-dedup. After H5 download + sequence extraction, run:
```
mmseqs easy-linclust sequences.fasta clust_res /tmp/mmseqs_tmp --min-seq-id 0.3 -c 0.8
```
Drop any duplicate (non-representative) and replace from same cell.
