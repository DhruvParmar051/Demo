"""
download_datasets.py
====================
Downloads government policy PDFs from four verified sources:

  1. IRS Publications       – irs.gov/pub/irs-pdf/
  2. SSA Publications       – ssa.gov/pubs/
  3. CMS Medicare Manuals   – cms.gov (Benefit Policy Manual, Pub 100-02)
  4. VA Federal Benefits    – va.gov
  5. FSA Handbook (2024-25) – fsapartners.ed.gov

All URLs were verified in April 2026.

Usage
-----
    pip install requests tqdm
    python download_datasets.py

Output layout
-------------
    data/raw_docs/
        irs/
        ssa/
        cms/
        va/
        fsa/

Each file is only downloaded once; re-running is safe (skips existing files).
A final summary prints total files downloaded and any failures.
"""

import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_BASE = Path("data/raw_docs")
TIMEOUT = 60          # seconds per request
RETRY_LIMIT = 3       # retries on transient errors
BACKOFF = 3           # seconds between retries
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AegisRAG-DataDownloader/1.0; "
        "+https://github.com/your-repo)"
    )
}

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each entry: (local_filename, url)

IRS_PUBS = [
    # Title                                          Pub number   Pages (approx)
    ("p17.pdf",    "https://www.irs.gov/pub/irs-pdf/p17.pdf"),     # Your Federal Income Tax ~300pp
    ("p15.pdf",    "https://www.irs.gov/pub/irs-pdf/p15.pdf"),     # Employer's Tax Guide
    ("p15a.pdf",   "https://www.irs.gov/pub/irs-pdf/p15a.pdf"),    # Employer's Supplemental Guide
    ("p15b.pdf",   "https://www.irs.gov/pub/irs-pdf/p15b.pdf"),    # Fringe Benefit Guide
    ("p501.pdf",   "https://www.irs.gov/pub/irs-pdf/p501.pdf"),    # Dependents & Standard Deduction
    ("p502.pdf",   "https://www.irs.gov/pub/irs-pdf/p502.pdf"),    # Medical & Dental Expenses
    ("p503.pdf",   "https://www.irs.gov/pub/irs-pdf/p503.pdf"),    # Child & Dependent Care
    ("p504.pdf",   "https://www.irs.gov/pub/irs-pdf/p504.pdf"),    # Divorced or Separated Individuals
    ("p505.pdf",   "https://www.irs.gov/pub/irs-pdf/p505.pdf"),    # Tax Withholding & Estimated Tax
    ("p525.pdf",   "https://www.irs.gov/pub/irs-pdf/p525.pdf"),    # Taxable & Nontaxable Income
    ("p526.pdf",   "https://www.irs.gov/pub/irs-pdf/p526.pdf"),    # Charitable Contributions
    ("p534.pdf",   "https://www.irs.gov/pub/irs-pdf/p534.pdf"),    # Depreciating Property Before 1987
    ("p535.pdf",   "https://www.irs.gov/pub/irs-pdf/p535.pdf"),    # Business Expenses
    ("p544.pdf",   "https://www.irs.gov/pub/irs-pdf/p544.pdf"),    # Sales & Other Dispositions of Assets
    ("p547.pdf",   "https://www.irs.gov/pub/irs-pdf/p547.pdf"),    # Casualties, Disasters & Thefts
    ("p550.pdf",   "https://www.irs.gov/pub/irs-pdf/p550.pdf"),    # Investment Income & Expenses
    ("p560.pdf",   "https://www.irs.gov/pub/irs-pdf/p560.pdf"),    # Retirement Plans (Small Business)
    ("p570.pdf",   "https://www.irs.gov/pub/irs-pdf/p570.pdf"),    # US Possessions Tax Guide
    ("p575.pdf",   "https://www.irs.gov/pub/irs-pdf/p575.pdf"),    # Pension & Annuity Income
    ("p590a.pdf",  "https://www.irs.gov/pub/irs-pdf/p590a.pdf"),   # Contributions to IRAs
    ("p590b.pdf",  "https://www.irs.gov/pub/irs-pdf/p590b.pdf"),   # Distributions from IRAs
    ("p596.pdf",   "https://www.irs.gov/pub/irs-pdf/p596.pdf"),    # Earned Income Credit
    ("p915.pdf",   "https://www.irs.gov/pub/irs-pdf/p915.pdf"),    # Social Security Benefits & Taxes
    ("p946.pdf",   "https://www.irs.gov/pub/irs-pdf/p946.pdf"),    # How to Depreciate Property
    ("p970.pdf",   "https://www.irs.gov/pub/irs-pdf/p970.pdf"),    # Tax Benefits for Education
    ("p463.pdf",   "https://www.irs.gov/pub/irs-pdf/p463.pdf"),    # Travel, Gift & Car Expenses
    ("p334.pdf",   "https://www.irs.gov/pub/irs-pdf/p334.pdf"),    # Small Business Tax Guide
    ("p587.pdf",   "https://www.irs.gov/pub/irs-pdf/p587.pdf"),    # Business Use of Your Home
    ("p936.pdf",   "https://www.irs.gov/pub/irs-pdf/p936.pdf"),    # Home Mortgage Interest
    ("p4681.pdf",  "https://www.irs.gov/pub/irs-pdf/p4681.pdf"),   # Cancelled Debts / Foreclosures
]

SSA_PUBS = [
    # Filename                 URL
    ("EN-05-10024.pdf", "https://www.ssa.gov/pubs/EN-05-10024.pdf"),  # Understanding the Benefits (~50pp)
    ("EN-05-10029.pdf", "https://www.ssa.gov/pubs/EN-05-10029.pdf"),  # Disability Benefits
    ("EN-05-10035.pdf", "https://www.ssa.gov/pubs/EN-05-10035.pdf"),  # Retirement Benefits
    ("EN-05-10069.pdf", "https://www.ssa.gov/pubs/EN-05-10069.pdf"),  # How Work Affects Your Benefits
    ("EN-05-10077.pdf", "https://www.ssa.gov/pubs/EN-05-10077.pdf"),  # What You Need to Know (Retirement)
    ("EN-05-10026.pdf", "https://www.ssa.gov/pubs/EN-05-10026.pdf"),  # Benefits for Children with Disabilities
    ("EN-05-10051.pdf", "https://www.ssa.gov/pubs/EN-05-10051.pdf"),  # Survivors Benefits
    ("EN-05-10137.pdf", "https://www.ssa.gov/pubs/EN-05-10137.pdf"),  # Payments While Outside the US
    ("EN-05-10153.pdf", "https://www.ssa.gov/pubs/EN-05-10153.pdf"),  # How You Earn Credits
    ("EN-05-10085.pdf", "https://www.ssa.gov/pubs/EN-05-10085.pdf"),  # Medicare
    ("EN-05-10043.pdf", "https://www.ssa.gov/pubs/EN-05-10043.pdf"),  # Medicare Savings Programs
    ("EN-05-10045.pdf", "https://www.ssa.gov/pubs/EN-05-10045.pdf"),  # Government Pension Offset
    ("EN-05-10007.pdf", "https://www.ssa.gov/pubs/EN-05-10007.pdf"),  # Government & Social Security
    ("EN-05-10018.pdf", "https://www.ssa.gov/pubs/EN-05-10018.pdf"),  # Identity Theft
    ("EN-64-030.pdf",   "https://www.ssa.gov/pubs/EN-64-030.pdf"),    # Red Book – Work Incentives (~100pp)
    ("EN-05-11000.pdf", "https://www.ssa.gov/pubs/EN-05-11000.pdf"),  # SSI (Supplemental Security Income)
    ("EN-05-11011.pdf", "https://www.ssa.gov/pubs/EN-05-11011.pdf"),  # SSI Spotlights
    ("EN-05-11015.pdf", "https://www.ssa.gov/pubs/EN-05-11015.pdf"),  # SSI & Students
    ("EN-05-11017.pdf", "https://www.ssa.gov/pubs/EN-05-11017.pdf"),  # Working While Disabled – PASS
    ("EN-05-10500.pdf", "https://www.ssa.gov/pubs/EN-05-10500.pdf"),  # SSA vs Medicare: Where to Go
]

# CMS Medicare Benefit Policy Manual (Pub 100-02)
# Direct PDF URLs confirmed from cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/
CMS_PUBS = [
    ("cms_bp102c01.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c01.pdf"),   # Ch 1  Inpatient Hospital (Part A)
    ("cms_bp102c02.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c02.pdf"),   # Ch 2  Inpatient Psychiatric Hospital
    ("cms_bp102c03.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c03.pdf"),   # Ch 3  Duration of Covered Inpatient Services
    ("cms_bp102c04.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c04.pdf"),   # Ch 4  Inpatient Psychiatric Benefit Days
    ("cms_bp102c05.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c05.pdf"),   # Ch 5  Lifetime Reserve Days
    ("cms_bp102c06.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c06.pdf"),   # Ch 6  Hospital Services Part B
    ("cms_bp102c07.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c07.pdf"),   # Ch 7  Home Health Services (~100pp)
    ("cms_bp102c08.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c08.pdf"),   # Ch 8  SNF Coverage (Extended Care)
    ("cms_bp102c09.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c09.pdf"),   # Ch 9  Hospice Services
    ("cms_bp102c10.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf"),   # Ch 10 Ambulance Services
    ("cms_bp102c11.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c11.pdf"),   # Ch 11 End Stage Renal Disease
    ("cms_bp102c12.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c12.pdf"),   # Ch 12 Durable Medical Equipment
    ("cms_bp102c13.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c13.pdf"),   # Ch 13 Rural Health Clinics
    ("cms_bp102c14.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c14.pdf"),   # Ch 14 Brachytherapy
    ("cms_bp102c15.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c15.pdf"),   # Ch 15 Covered Medical Services
    ("cms_bp102c16.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c16.pdf"),   # Ch 16 General Exclusions
    ("cms_bp102c17.pdf",  "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c17.pdf"),   # Ch 17 Drugs & Biologicals
]

VA_PUBS = [
    (
        "va_federal_benefits_2025.pdf",
        "https://department.va.gov/wp-content/uploads/2024/12/2025-Federal-Benefits-for-Veterans-Dependents-and-Survivors.pdf",
    ),
]

# FSA Handbook 2024-2025 – confirmed direct PDFs from fsapartners.ed.gov
_FSA_BASE = (
    "https://fsapartners.ed.gov/sites/default/files/2024-2025"
    "/2024-2025_Federal_Student_Aid_Handbook"
    "/_knowledge-center_fsa-handbook_2024-2025"
)
FSA_PUBS = [
    ("fsa_avguide.pdf", f"{_FSA_BASE}_application-and-verification-guide.pdf"),  # App & Verification Guide
    ("fsa_vol1.pdf",    f"{_FSA_BASE}_vol1.pdf"),   # Student Eligibility
    ("fsa_vol2.pdf",    f"{_FSA_BASE}_vol2.pdf"),   # School Eligibility & Operations
    ("fsa_vol3.pdf",    f"{_FSA_BASE}_vol3.pdf"),   # Academic Calendars, COA, Packaging
    ("fsa_vol4.pdf",    f"{_FSA_BASE}_vol4.pdf"),   # Processing Aid & Managing FSA Funds
    ("fsa_vol5.pdf",    f"{_FSA_BASE}_vol5.pdf"),   # Withdrawals & Return of Title IV Funds
    ("fsa_vol6.pdf",    f"{_FSA_BASE}_vol6.pdf"),   # Campus-Based Programs
    ("fsa_vol7.pdf",    f"{_FSA_BASE}_vol7.pdf"),   # The Federal Pell Grant Program
    ("fsa_vol8.pdf",    f"{_FSA_BASE}_vol8.pdf"),   # The Direct Loan Program
    ("fsa_vol9.pdf",    f"{_FSA_BASE}_vol9.pdf"),   # The TEACH Grant Program
]

DATASETS = [
    ("irs", IRS_PUBS),
    ("ssa", SSA_PUBS),
    ("cms", CMS_PUBS),
    ("va",  VA_PUBS),
    ("fsa", FSA_PUBS),
]

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def download_file(url: str, dest: Path) -> tuple[bool, str]:
    """Download *url* to *dest*.  Returns (success, message)."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True, f"skipped (already exists, {_human_size(dest.stat().st_size)})"

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            if resp.status_code == 404:
                return False, f"HTTP 404 – URL not found: {url}"
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower() and "pdf" not in content_type.lower():
                # Government sites sometimes return an HTML error page with 200 OK
                return False, f"Got HTML instead of PDF (Content-Type: {content_type})"

            total = int(resp.headers.get("Content-Length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)

            with open(dest, "wb") as fh:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

            size = dest.stat().st_size
            if size < 1024:
                dest.unlink(missing_ok=True)
                return False, f"File too small ({size} bytes) – likely an error page"

            return True, f"OK ({_human_size(size)})"

        except requests.exceptions.Timeout:
            msg = f"Timeout (attempt {attempt}/{RETRY_LIMIT})"
        except requests.exceptions.ConnectionError as exc:
            msg = f"Connection error: {exc} (attempt {attempt}/{RETRY_LIMIT})"
        except requests.exceptions.RequestException as exc:
            msg = f"Request error: {exc} (attempt {attempt}/{RETRY_LIMIT})"

        if attempt < RETRY_LIMIT:
            time.sleep(BACKOFF * attempt)

    return False, msg  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_ok = 0
    total_skip = 0
    failures: list[tuple[str, str]] = []

    for source_name, file_list in DATASETS:
        dest_dir = OUTPUT_BASE / source_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {source_name.upper()} — {len(file_list)} files → {dest_dir}")
        print(f"{'='*60}")

        for filename, url in tqdm(file_list, desc=source_name, unit="file"):
            dest = dest_dir / filename
            ok, msg = download_file(url, dest)

            if not ok:
                tqdm.write(f"  ✗ {filename}: {msg}")
                failures.append((f"{source_name}/{filename}", msg))
            elif "skipped" in msg:
                tqdm.write(f"  ↩ {filename}: {msg}")
                total_skip += 1
            else:
                tqdm.write(f"  ✓ {filename}: {msg}")
                total_ok += 1

            # Be polite to government servers – small delay between requests
            time.sleep(0.4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded : {total_ok}")
    print(f"  Skipped    : {total_skip}  (already on disk)")
    print(f"  Failed     : {len(failures)}")

    if failures:
        print("\n  FAILED FILES:")
        for name, reason in failures:
            print(f"    - {name}")
            print(f"        {reason}")
        print(
            "\n  Tip: Failed files are usually 404s because a pub has been\n"
            "  renumbered or moved. Check the source URL in a browser and\n"
            "  update the URL in this script if needed."
        )

    print(f"\n  Output directory: {OUTPUT_BASE.resolve()}")
    print("  Done.\n")


if __name__ == "__main__":
    main()