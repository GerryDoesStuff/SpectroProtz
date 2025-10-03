def write_workbook(out_path: str, processed, qc_table, audit, figures):
    """Placeholder: Implement openpyxl-based export here."""
    # For scaffold only, just write a tiny text file indicating where Excel would go
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Workbook would be written here.\n")
        f.write(f"Processed spectra: {len(processed)}\n")
        f.write(f"QC rows: {len(qc_table)}\n")
        f.write("\nAudit:\n" + "\n".join(audit))
    return out_path
