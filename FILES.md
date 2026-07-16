# Repository update files

## Scripts

- `scripts/08_figures/make_named_regional_cardiac_heatmaps.py`
  - Calculates regional FA–cardiac and normalized regional volume–cardiac Spearman associations.
  - Uses the corrected 26-mouse cardiac mapping.
  - Applies BH-FDR correction and produces named ROI heatmaps.

- `scripts/08_figures/redraw_paired_regional_heatmaps.py`
  - Produces simplified publication/poster heatmaps.
  - Uses a common color scale and global BH-FDR significance markers.

## Documentation

- `docs/ROI_mapping_332_named.csv`
  - Maps 332 unilateral ROI indices to CHASS atlas labels, abbreviations, anatomical names, and ontology levels.
  - Atlas hemisphere blocks are currently denoted A/B because anatomical left/right assignment remains unresolved.

## Figures

- Regional FA × cardiac phenotypes
- Normalized regional volume × cardiac phenotypes
- Paired PDF
