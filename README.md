# Imaging the Heart–Brain Connection and Metabolic Modulation

This repository contains reproducible code for processing and analyzing GLP-1 mouse brain imaging data, including diffusion MRI preprocessing, tractography, atlas-based structural connectomes, regional diffusion metrics, regional brain volumes, metadata integration, quality control, and exploratory heart–brain association analyses.

## Current cohort and data status

The current analysis-ready imaging cohort contains **26 mice**.

Confirmed MRI-to-cardiac mappings include:

| MRI ID | Badea ID | Cardiac ID |
|---|---:|---:|
| B26010604 | 250530_13 | 250530-13 |
| B26010605 | 250530_14 | 250530-14 |
| B26010606 | 250530_15 | 250530-15 |

Current cardiac availability:

- Most cardiac phenotypes: **n = 26**
- RV stroke volume and myocardial mass: **n = 25**
- Body weight remains missing for B26010605 and B26010606; weight-adjusted models therefore use fewer observations unless weight is restored from a validated source.

The canonical mouse-level metadata/cardiac table is maintained outside GitHub as `GLP1_MRI.csv`. Raw imaging files, derived NIfTI/MIF/TCK files, protected metadata, and analysis outputs are not committed to this repository.

## Main pipeline

1. DWI preprocessing and T2-to-b0 registration
2. T2-mask-based tensor metrics: FA, MD, AD, and RD
3. 100k streamline tractography
4. Atlas-label orientation and native-space registration
5. Atlas-based count and length connectomes
6. Regional diffusion metrics
7. Regional absolute and normalized brain volumes
8. Metadata and cardiac-data integration
9. Quality control and missing-data summaries
10. Brain–heart association analyses and figures

## ROI definition and anatomical names

The analysis uses **332 unilateral atlas ROIs**:

- raw labels `1–166`
- raw labels `1001–1166`

These correspond to 166 homologous anatomical structures. Anatomical names are obtained from the CHASS N51200 ontology file `civm_mouse_v2_ontology.csv`. Despite its extension, this file is tab-separated.

The two raw-label blocks are currently represented as **hemisphere block A** and **hemisphere block B**. Anatomical left/right assignment has not yet been verified and should not be inferred from the label values alone.

Use `atlas_abbrev + anatomical_name` as the unique structure display label because two distinct ontology entries share the readable name “Pontine Reticular Nucleus.”

Generate the named mapping with:

```bash
python scripts/05_metadata/build_named_roi_mapping.py \
  --ontology /path/to/civm_mouse_v2_ontology.csv \
  --base-mapping /path/to/ROI_mapping_332.csv \
  --output /path/to/ROI_mapping_332_named.csv
```

## Current regional brain–heart findings

Using Spearman correlations and global Benjamini–Hochberg FDR correction across all ROI × cardiac tests within each imaging modality:

- Vestibular nuclei FA was positively associated with ejection fraction.
- Hypothalamic FA was negatively associated with LV end-systolic volume.
- Normalized regional brain volume showed coordinated spatial patterns, but no volume association remained significant after global modality-wide FDR correction.

These are cross-sectional associations and do not establish causality.

Generate the named publication-style heatmaps with:

```bash
python scripts/08_figures/make_regional_cardiac_heatmaps.py \
  --brain-features /path/to/advanced_brain_heart_merged_features.csv \
  --master-table /path/to/GLP1_MRI.csv \
  --roi-mapping /path/to/ROI_mapping_332_named.csv \
  --output-dir /path/to/regional_cardiac_heatmaps
```

## Repository structure

```text
scripts/
├── 01_preprocessing/
├── 02_connectome/
├── 03_regional_metrics/
├── 04_global_connectome/
├── 05_metadata/
│   └── build_named_roi_mapping.py
├── 06_analysis_ready/
├── 07_statistics/
├── 08_figures/
│   └── make_regional_cardiac_heatmaps.py
└── archive/

docs/
└── current_analysis_status.md
```

## Statistical reporting

The heatmap script reports two FDR quantities:

- `q_by_outcome`: BH-FDR across 332 ROIs for one cardiac phenotype; useful for exploratory visualization.
- `q_global_modality`: BH-FDR across all ROI × cardiac tests within FA or normalized volume; used for formal significance stars in publication-style figures.

## Data policy

Do not commit:

- raw or derived imaging volumes
- tractography files
- mouse-level protected metadata
- cardiac source tables
- large generated CSV outputs
- logs containing infrastructure paths or identifiers

The repository should contain code, documentation, and small non-sensitive reference files only.

## License

MIT License.
