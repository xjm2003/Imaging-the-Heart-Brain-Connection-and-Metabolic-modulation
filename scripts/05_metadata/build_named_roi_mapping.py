#!/usr/bin/env python3
"""Build a named 332-ROI mapping from the CHASS mouse ontology.

The CHASS ontology file has a .csv extension but is tab-separated. The
analysis atlas uses raw labels 1-166 and 1001-1166. The second label block is
mapped to ontology values 1-166 by subtracting 1000.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--base-mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def clean_structure_name(value: str) -> str:
    value = str(value).strip()
    if "__" in value:
        value = value.split("__", 1)[1]
    return value.replace("_", " ").strip()


def main() -> None:
    args = parse_args()

    ontology = pd.read_csv(
        args.ontology,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    ontology.columns = ontology.columns.str.strip()

    required = {
        "Value", "Structure", "Abbrev", "c_R", "c_G", "c_B", "c_A",
        "Level_1", "Level_2", "Level_3", "Level_4",
    }
    missing = sorted(required - set(ontology.columns))
    if missing:
        raise ValueError(f"Ontology is missing required columns: {missing}")

    ontology["Value"] = pd.to_numeric(
        ontology["Value"], errors="raise"
    ).astype(int)

    expected_ontology = set(range(1, 167))
    observed_ontology = set(ontology["Value"])
    if observed_ontology != expected_ontology:
        raise ValueError(
            "Ontology labels are not exactly 1-166. "
            f"Missing={sorted(expected_ontology - observed_ontology)}, "
            f"extra={sorted(observed_ontology - expected_ontology)}"
        )

    ontology["anatomical_name"] = ontology["Structure"].map(
        clean_structure_name
    )
    ontology_lookup = ontology.rename(
        columns={
            "Value": "ontology_value",
            "Structure": "atlas_structure",
            "Abbrev": "atlas_abbrev",
            "c_R": "atlas_color_R",
            "c_G": "atlas_color_G",
            "c_B": "atlas_color_B",
            "c_A": "atlas_color_A",
            "Level_1": "ontology_level_1",
            "Level_2": "ontology_level_2",
            "Level_3": "ontology_level_3",
            "Level_4": "ontology_level_4",
        }
    )[
        [
            "ontology_value", "atlas_structure", "atlas_abbrev",
            "anatomical_name", "atlas_color_R", "atlas_color_G",
            "atlas_color_B", "atlas_color_A", "ontology_level_1",
            "ontology_level_2", "ontology_level_3", "ontology_level_4",
        ]
    ]

    mapping = pd.read_csv(args.base_mapping)
    mapping["raw_atlas_label_id"] = pd.to_numeric(
        mapping["raw_atlas_label_id"], errors="raise"
    ).astype(int)

    expected_raw = set(range(1, 167)) | set(range(1001, 1167))
    observed_raw = set(mapping["raw_atlas_label_id"])
    if observed_raw != expected_raw:
        raise ValueError(
            "Base mapping must contain raw labels 1-166 and 1001-1166. "
            f"Missing={sorted(expected_raw - observed_raw)}, "
            f"extra={sorted(observed_raw - expected_raw)}"
        )

    mapping["ontology_value"] = mapping["raw_atlas_label_id"].map(
        lambda value: value if value <= 166 else value - 1000
    )
    mapping["hemisphere"] = mapping["raw_atlas_label_id"].map(
        lambda value: "hemisphere_block_A" if value <= 166
        else "hemisphere_block_B"
    )

    mapping = mapping.drop(columns=["anatomical_name"], errors="ignore")
    named = mapping.merge(
        ontology_lookup,
        on="ontology_value",
        how="left",
        validate="many_to_one",
    )

    if len(named) != 332 or named["ROI_index"].nunique() != 332:
        raise ValueError("Named mapping must contain 332 unique ROI indices.")
    if named["anatomical_name"].isna().any():
        raise ValueError("At least one ROI could not be assigned a name.")

    named["structure_display_name"] = (
        named["atlas_abbrev"].astype(str)
        + " — "
        + named["anatomical_name"].astype(str)
    )
    block = named["hemisphere"].map(
        {"hemisphere_block_A": "A", "hemisphere_block_B": "B"}
    )
    named["roi_unique_name"] = block + " | " + named["structure_display_name"]

    if named["roi_unique_name"].nunique() != 332:
        raise ValueError("ROI unique names are not unique across 332 rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    named.to_csv(args.output, index=False)

    base = named.drop_duplicates("ontology_value")
    print(f"Saved: {args.output}")
    print(f"Rows: {len(named)}")
    print(f"Ontology structures: {base['ontology_value'].nunique()}")
    print(f"Unique abbreviations: {base['atlas_abbrev'].nunique()}")
    print(f"Unique cleaned names: {base['anatomical_name'].nunique()}")
    print("Hemisphere blocks remain A/B until anatomical laterality is verified.")


if __name__ == "__main__":
    main()
