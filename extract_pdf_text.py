import fitz

for pdf_name in [
    "Reddit 1.pdf",
    "Reddit 2.pdf",
    "Reddit 3.pdf",
    "Reddit 4.pdf",
]:
    path = f"inputs/{pdf_name}"

    document = fitz.open(path)

    print(f"\n\n===== {pdf_name} =====\n")

    for page in document:
        print(page.get_text())