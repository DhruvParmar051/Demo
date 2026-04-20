# Training Data Sources for AegisRAG

All sources below are **freely downloadable**, mostly U.S. government or permissively licensed corporate documentation, and structurally match what a real customer-support KB looks like (policies, procedures, FAQs, eligibility rules). Drop them into `data/raw/<domain>/` and run `python run.py ingest --source-dir data/raw/`.

Tier 1 = start here, highest signal for CGAL / citation training.
Tier 2 = add once Tier 1 is flowing.
Tier 3 = vertical-specific expansion.

---

## Tier 1 — core corpus (do these first)

### 1. MultiDoc2Dial
- **URL:** https://doc2dial.github.io/multidoc2dial/
- **License:** CC-BY-SA 4.0
- **Size:** ~4.8K documents, 488K dialogue turns, 4 domains (DMV, VA, SSA, StudentAid)
- **Format:** JSON (contains the source docs and grounded dialogue turns)
- **Why:** This is the dataset the system was designed against. Every plan benchmark assumes it. Converts to TXT with a 20-line script (see `scripts/prepare_multidoc2dial.py` — write this on first use).
- **Download:**
  ```bash
  wget https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip -P data/raw/
  unzip data/raw/multidoc2dial.zip -d data/raw/multidoc2dial
  ```

### 2. IRS Publications (hundreds of tax-support PDFs)
- **URL:** https://www.irs.gov/forms-pubs
- **License:** public domain (U.S. government work)
- **Size:** 200+ PDFs, 50-400 pages each
- **Why:** Dense policy prose with exact citations — ideal for training the citation-weighted SFT loss. Lots of "if X then Y" procedural text that exercises query decomposition.
- **Key files to pull:**
  - Publication 17 — Your Federal Income Tax (~280 pp)
  - Publication 15 — Employer's Tax Guide
  - Publication 501 — Dependents, Standard Deduction
  - Publication 463 — Travel, Gift, Car Expenses
  - Publication 525 — Taxable and Nontaxable Income
  - Publication 590-A / 590-B — IRAs
  - Publication 970 — Tax Benefits for Education
- **Bulk download:**
  ```bash
  mkdir -p data/raw/irs
  for pub in 17 15 501 463 525 590a 590b 970 535 334; do
    wget "https://www.irs.gov/pub/irs-pdf/p${pub}.pdf" -P data/raw/irs/
  done
  ```

### 3. Social Security Administration (SSA) Publications
- **URL:** https://www.ssa.gov/pubs/
- **License:** public domain
- **Why:** Matches one of MultiDoc2Dial's domains; gives you depth for that vertical.
- **Key files:**
  - "Understanding Supplemental Security Income" (EN-17-008)
  - "What You Need to Know When You Get Retirement or Survivors Benefits" (EN-05-10077)
  - "Disability Benefits" (EN-05-10029)
  - "Red Book — Summary Guide to Employment Support" (EN-64-030)
- **Download:** manual pick from the pubs index (URLs change); ~40 PDFs total.

### 4. VA Benefits Books
- **URL:** https://www.va.gov/opa/publications/benefits_book/
- **License:** public domain
- **Files:** "Federal Benefits for Veterans, Dependents, and Survivors" (annual, 120+ pp)
- **Why:** Second domain that overlaps MultiDoc2Dial; lets you stratify eval by domain.

### 5. California DMV Handbook
- **URL:** https://www.dmv.ca.gov/portal/handbook/
- **Files:** California Driver Handbook (HTML + PDF), CA Commercial Driver Handbook, CA Motorcycle Handbook
- **Why:** Short, FAQ-style, lots of numbered rules. Good signal for policy-type question generation.
- **Download:**
  ```bash
  mkdir -p data/raw/dmv
  wget https://www.dmv.ca.gov/portal/file/california-driver-handbook-pdf/ -O data/raw/dmv/ca_driver_handbook.pdf
  wget https://www.dmv.ca.gov/portal/file/california-commercial-driver-handbook-pdf/ -O data/raw/dmv/ca_cdl_handbook.pdf
  ```

---

## Tier 2 — additional corporate / tech support (for domain breadth)

### 6. PostgreSQL Manual
- **URL:** https://www.postgresql.org/docs/current/pdf/
- **License:** PostgreSQL License (BSD-like)
- **Files:** postgresql-A4.pdf (~3000 pp single document)
- **Why:** Dense technical reference with sub-sections — good stress test for chunking span offsets.

### 7. Kubernetes Documentation
- **URL:** https://github.com/kubernetes/website
- **License:** CC-BY-4.0
- **Files:** `content/en/docs/**/*.md` — clone and `find . -name '*.md'` gives ~2500 Markdown files
- **Download:**
  ```bash
  git clone --depth 1 https://github.com/kubernetes/website data/raw/k8s-docs
  find data/raw/k8s-docs/content/en/docs -name '*.md' | head
  ```

### 8. AWS Whitepapers
- **URL:** https://aws.amazon.com/whitepapers/
- **License:** Amazon content; re-distribute restricted but fair-use for training in-house models is generally accepted
- **Recommended:**
  - AWS Well-Architected Framework (6 pillars, PDFs)
  - AWS Security Best Practices
  - Serverless Applications Lens
- **Why:** Excellent structured policy prose with "do / don't / why" patterns.

### 9. Microsoft Learn / Docs (as Markdown)
- **URL:** https://github.com/MicrosoftDocs
- **License:** CC-BY-4.0 for docs under `MicrosoftDocs/` umbrella
- **Suggested repos:**
  - `MicrosoftDocs/azure-docs`
  - `MicrosoftDocs/365-docs`
  - `MicrosoftDocs/windows-itpro-docs`
- **Download:**
  ```bash
  git clone --depth 1 https://github.com/MicrosoftDocs/azure-docs data/raw/azure-docs
  ```

### 10. Ubuntu Documentation
- **URL:** https://help.ubuntu.com/ (HTML), `man` pages (TXT)
- **License:** CC-BY-SA
- **Why:** Procedural troubleshooting prose — the exact genre customer support handles.

---

## Tier 3 — vertical-specific packs (pick one based on your target customer)

### Healthcare / life-sciences
- **MedlinePlus** (NIH) — https://medlineplus.gov/xml.html (XML data dumps, public domain)
- **CMS Medicare Publications** — https://www.cms.gov/medicare
- **FDA Consumer Updates** — https://www.fda.gov/consumers
- **Why:** If your acquirer is a health insurer or telehealth company, this is your home field.

### Financial services
- **SEC Investor Bulletins** — https://www.sec.gov/investor/pubs.shtml
- **FINRA Rulebook** — https://www.finra.org/rules-guidance/rulebooks
- **CFPB Consumer Advisories** — https://www.consumerfinance.gov/consumer-tools/
- **Federal Reserve Consumer Compliance Handbook** — PDFs on federalreserve.gov

### Legal
- **CUAD — Contract Understanding Atticus Dataset** — https://www.atticusprojectai.org/cuad (CC-BY-4.0, 510 contracts)
- **U.S. Code** (public domain) — https://uscode.house.gov/download/download.shtml
- **Federal Register / Code of Federal Regulations** — https://www.govinfo.gov/

### Education
- **Federal Student Aid Handbook** — https://fsapartners.ed.gov/knowledge-center/fsa-handbook (annually updated, 1000+ pp PDF — matches MultiDoc2Dial's StudentAid domain)
- **Department of Education publications** — https://www.ed.gov/publications

### E-commerce / retail
- **AmazonQA** — https://github.com/amazonqa/amazonqa (1.4M Q&A pairs, Amazon product questions — great for preference-triplet mining)

### Government / public services
- **USA.gov Help** — https://www.usa.gov/
- **USCIS Policy Manual** — https://www.uscis.gov/policy-manual (immigration customer support, dense policy language)
- **IRS Internal Revenue Manual** — https://www.irs.gov/irm (the handbook IRS agents themselves use)

---

## Synthetic-data seed corpora (for training the generators themselves)

These aren't for ingestion; they're for fine-tuning the teacher model to produce high-quality synthetic QA:

### Stack Exchange Data Dump
- **URL:** https://archive.org/details/stackexchange
- **License:** CC-BY-SA
- **Size:** ~200 GB compressed, hundreds of sites. For customer-support tone, focus on:
  - `superuser.com`
  - `askubuntu.com`
  - `serverfault.com`
  - `webapps.stackexchange.com`
- **Why:** Real user-support Q&A threads with accepted-answer labels — perfect for training preference pairs.

### Ubuntu Dialogue Corpus
- **URL:** https://github.com/rkadlec/ubuntu-ranking-dataset-creator
- **Size:** 1M dialogues
- **Why:** Genuine multi-turn customer support conversations.

### DoQA (Domain Question Answering)
- **URL:** https://ixa2.si.ehu.eus/convai/
- **License:** CC-BY-SA
- **Why:** Crowdsourced domain-grounded Q&A in cooking, travel, movies — helps generalize beyond gov/tax domains.

---

## Recommended starter bundle (minimal — one afternoon to assemble)

If you want to reproduce the plan's numbers with the least effort, pull exactly:

```
data/raw/multidoc2dial/    (Tier 1 #1)   ~4800 source docs
data/raw/irs/              (Tier 1 #2)   10 flagship PDFs, ~2000 pp total
data/raw/dmv/              (Tier 1 #5)   3 PDFs
data/raw/postgresql/       (Tier 2 #6)   1 massive PDF for stress-testing chunker
data/raw/k8s-docs/         (Tier 2 #7)   ~2500 Markdown files
```

That's roughly **~6K documents, ~50K chunks after splitting** — matches the plan's ingestion target and gives four-way domain stratification for evaluation.

---

## Legal note

For anything beyond personal research / academic use, **verify the license per document** before including it in a product dataset. Government publications (IRS, SSA, VA, DMV, USA.gov) are public domain in the U.S. but may have restrictions in other jurisdictions. CC-BY-SA sources require attribution and may propagate copyleft to derived models — talk to a lawyer before you ship a commercial product trained on them.
