import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = Path(".")  # MedQuAD repo root

OUTPUT_CSV = ROOT / "medquad_raw.csv"

# Folders in MedQuAD that contain QA XML files
QA_DIRS = [
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
]

def parse_xml_file(path: Path, source: str):
    """
    Parse a single MedQuAD XML file and return a list of dicts
    with question, answer, url, and source.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    records = []
    # The exact tags can vary slightly, but many MedQuAD files use
    # <document> or similar. We look for question/answer/url fields.
    q_text = None
    a_text = None
    url = None

    # Try to find obvious tags
    # Adjust tags if needed after inspecting a few XMLs
    q_node = root.find(".//question")
    a_node = root.find(".//answer")
    url_node = root.find(".//url")

    if q_node is not None:
        q_text = (q_node.text or "").strip()
    if a_node is not None:
        a_text = (a_node.text or "").strip()
    if url_node is not None:
        url = (url_node.text or "").strip()

    if q_text and a_text:
        records.append({
            "question": q_text,
            "answer": a_text,
            "url": url or "",
            "source": source
        })

    return records

def main():
    all_rows = []
    for d in QA_DIRS:
        dir_path = ROOT / d
        if not dir_path.exists():
            continue
        source = d  # use folder name as source label
        for fname in os.listdir(dir_path):
            if not fname.endswith(".xml"):
                continue
            fpath = dir_path / fname
            try:
                recs = parse_xml_file(fpath, source)
                all_rows.extend(recs)
            except Exception as e:
                print("Error parsing", fpath, ":", e)

    df = pd.DataFrame(all_rows)
    print("Total QA pairs parsed:", len(df))
    df.to_csv(OUTPUT_CSV, index=False)
    print("Wrote CSV to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
