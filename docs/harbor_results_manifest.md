# Harbor analysis files

This folder records Harbor-provided analysis notebooks and result categories.

## Full cohort / cognition-cardiac analysis

Source package: Full_cohort.zip

Main notebook:

- 
otebooks/harbor_full_cohort/Cohort_Cog_Cardiac_Analysis_v3_all23.ipynb

Associated data/results in the source package included:

- MWM all-mouse day-level table
- MWM-cardiac matched table
- cardiac/MWM matched datasheet
- learning, memory, cardiac, nonlinear, treatment-moderation, and correlation figures

The source CSV/XLSX tables are not committed to this repository.

## Mass / body-weight analysis

Source package: mass.zip

Main notebooks:

- 
otebooks/harbor_mass/1GLP1_Cohort_Mass_Analysis_v3_glp1base.ipynb
- 
otebooks/harbor_mass/1GLP1_Mass_NLME_v2_glp1base.ipynb

Associated data/results in the source package included:

- GLP-1 mass analysis result tables
- NLME result tables
- mouse body-weight source table
- trajectory, sex effect, genotype effect, genotype-by-sex interaction, LME, and NLME diagnostic figures

The source XLSX result and data tables are not committed to this repository.

## Relationship to current imaging pipeline

Harbor's notebooks are related to the larger GLP-1 cohort, cognition/MWM outcomes, cardiac variables, and body-mass trajectories. They are separate from the main DWI/T2/tractography/connectome pipeline.

The current imaging pipeline produces regional FA/MD/AD/RD, regional volume, connectomes, QC summaries, and imaging-cardiac overlap tables. Harbor's analyses may provide external cognitive, cardiac, treatment, and mass variables for later integration.
