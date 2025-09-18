# pdf2csv

Utility scripts for extracting text from PDF documents.

## Requirements
- Python 3.10+
- [`pdfplumber`](https://github.com/jsvine/pdfplumber): `pip install pdfplumber`

## Usage
Run the extractor directly to convert a PDF into a Markdown file where each
page is saved under its own heading. The script scans the opening pages for a
table of contents: any detected TOC lines are normalised into a Markdown
bullet list at the top of the output, and pages before the TOC are dropped.

```bash
python extract_text.py path/to/document.pdf
```

By default the script writes `path/to/document.md`, removes the top 5% and
bottom 5% of every page to drop fixed headers/footers, and automatically keeps
table-of-contents pages separate from the main body. You can fine-tune this
behaviour and other parsing settings with the available options:

- `--header-cutoff FLOAT` – Fraction of the page height to ignore at the top.
  Use `0` to disable header filtering. Default: `0.05` (top 5%).
- `--footer-cutoff FLOAT` – Fraction of the page height to ignore at the bottom.
  Use `1` to disable footer filtering. Default: `0.95` (keep everything above
  the bottom 5%).
- `--x-tolerance FLOAT` / `--y-tolerance FLOAT` – Passed through to
  `pdfplumber` when clustering words into lines.
- `--max-pages INT` – Only parse the first *N* pages. Helpful for debugging.
- `--preview` – Print the extracted text for each processed page to stdout
  (TOC entries are shown first when detected).
- `--output PATH` – Write the Markdown result to a specific location (defaults
  to replacing the PDF extension with `.md`).

### Debugging headers/footers
While adjusting the header/footer cutoffs, combine `--preview` with
`--max-pages` to quickly inspect the effect on the first few pages:

```bash
python extract_text.py report.pdf --max-pages 3 --preview --header-cutoff 0.04 --footer-cutoff 0.9
```

Once satisfied, rerun without `--max-pages` to export the entire file.