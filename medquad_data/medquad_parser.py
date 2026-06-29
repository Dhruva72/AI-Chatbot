"""
medquad_parser.py
Parses MedQuAD XML files into a flat CSV (medquad_index.csv).
Run once: python medquad_parser.py
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd

# Folders 10, 11, 12 have answers removed due to copyright - skip them.
SKIP_FOLDERS = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}

def parse_medquad(dataset_dir, output_csv="medquad_index.csv"):
    """
    Walk all valid MedQuAD subfolders, extract QA pairs, save to CSV.

    Args:
        dataset_dir: Path to the cloned MedQuAD repo root
        output_csv:  Where to save the resulting CSV
    """
    records = []

    for folder in sorted(os.listdir(dataset_dir)):
        if folder in SKIP_FOLDERS:
            print("  [SKIP] " + folder + " (no answers)")
            continue

        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
        print("  [OK]   " + folder + "  (" + str(len(xml_files)) + " files)")

        for xml_file in xml_files:
            filepath = os.path.join(folder_path, xml_file)
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()

                for qa in root.iter("QAPair"):
                    q_el = qa.find("Question")
                    a_el = qa.find("Answer")

                    if q_el is None or a_el is None:
                        continue

                    question = (q_el.text or "").strip()
                    answer   = (a_el.text or "").strip()

                    if not question or not answer:
                        continue

                    focus = root.findtext("Focus", default="").strip()
                    qtype = q_el.get("qtype", "").strip()
                    source = folder

                    records.append({
                        "question": question,
                        "answer":   answer,
                        "focus":    focus,
                        "qtype":    qtype,
                        "source":   source,
                    })

            except ET.ParseError as e:
                print("    [WARN] Could not parse " + xml_file + ": " + str(e))

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["question"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print("\nDone! Saved " + str(len(df)) + " QA pairs -> " + output_csv)
    return df


if __name__ == "__main__":
    import sys

    DATA_DIR = r"D:\project\NLP\medquad_data"
    OUT_CSV  = r"D:\project\NLP\medquad_index.csv"

    if not os.path.isdir(DATA_DIR):
        print("ERROR: Dataset folder not found: " + DATA_DIR)
        print("Clone the repo first:  git clone https://github.com/abachaa/MedQuAD medquad_data")
        sys.exit(1)

    print("Parsing MedQuAD from: " + DATA_DIR)
    parse_medquad(DATA_DIR, OUT_CSV)