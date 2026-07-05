# Imaging the Heart-Brain Connection and Metabolic Modulation

This repository contains scripts for processing and analyzing GLP-1 mouse brain imaging data, including diffusion MRI preprocessing, tractography, atlas-based structural connectome construction, regional diffusion metrics, regional brain volumes, metadata integration, QC, and exploratory heart-brain association analyses.

## Current cohort structure

The current imaging cohort includes 26 mice after adding B25122904.

- The original 25-mouse analysis-ready package is retained as a frozen version.
- B25122904 was added later as an imaging + metadata case.
- B25122904 has DWI, T2, atlas label, metadata, tractography, and connectome outputs.
- B25122904 is not currently present in the cardiac/MWM merged table, so cardiac analyses use the available cardiac-matched subset.

## Main pipeline

1. DWI preprocessing and T2-to-b0 registration
2. T2-mask-based tensor metrics: FA, MD, AD, RD
3. 100k tractography
4. Atlas label auto-orientation
5. Atlas-based count and mean-FA connectomes
6. Regional diffusion metrics
7. Regional absolute and normalized brain volumes
8. Metadata and cardiac/MWM merge
9. QC, overlap summary, missing-data summary
10. Genotype, sex, cardiac association, and exploratory treatment analyses

## Repository structure

```text
scripts/
├── 01_preprocessing/
├── 02_connectome/
├── 03_regional_metrics/
├── 04_global_connectome/
├── 05_metadata/
├── 06_analysis_ready/
├── 07_statistics/
├── 08_figures/
└── archive/
```

## Data policy

Raw imaging files, derived NIfTI/MIF/TCK files, CSV outputs, metadata source files, and logs are not stored in this repository.

