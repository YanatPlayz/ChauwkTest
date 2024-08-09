from img2table.ocr import TesseractOCR
from img2table.document import PDF

# Instantiation of OCR
ocr = TesseractOCR(n_threads=1, lang="eng")

# Instantiation of document, either an image or a PDF
doc = PDF("data/Model_Career_Centers.pdf")

# Table extraction
extracted_tables = doc.extract_tables(ocr=ocr, implicit_rows=False, borderless_tables=False, min_confidence=50)

print(extracted_tables)

for key, value_list in extracted_tables.items():
    for value in value_list:
        table_df = value.df
        print(f"Table {key}:")
        print(table_df)
        print("\n")