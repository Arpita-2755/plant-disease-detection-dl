# Submission checklist

## Deliverables

- [x] **Working demo** - `streamlit run app/app.py` (trained models included in `models/`)
- [x] **GitHub repository** - committed locally with logical history, ready to push
- [ ] **Push to GitHub** - needs the owner's login (commands below)
- [x] **Report** - `report/Arpita_Mishra_12303670_INT422_Project_Report.docx` and `.pdf`

## Verified

- [x] Clean virtual environment: `pip install -r requirements.txt` installs and imports
- [x] `python -m src.prepare_data` downloads the subset and matches all checksums
- [x] all stages of `python -m src.run_all` completed from that clean environment and produced the reported numbers (the run was paused once and resumed at the autoencoder stage)
- [x] App started from the clean environment and driven by Playwright without exceptions
- [x] Fresh `git clone`: app test passes for 3 models x 10 samples (`python tests/test_app.py`)
- [x] Two full classifier runs in separate environments gave identical results
- [x] TensorBoard shown from the project `.venv` (Smart App Control blocked its `grpc` library in the temporary clean copy)
- [x] Report numbers cross-checked against `metrics.json` (`python tools/verify_submission.py`)
- [x] Citations checked against Crossref, arXiv or the publisher
- [x] Report pages rasterised and inspected

## Known limitations (also stated in the report)

- Docker image not built: Docker is not installed on the development machine
- The PDF was exported with Microsoft Word because LibreOffice is not installed
- Dataset fetched from the authors' Hugging Face mirror, not from tensorflow_datasets
- Training took 55 minutes in the final run, above the 45-minute target
- The report has 49 numbered pages plus 7 of front matter
- No GPU or DGX hardware was used; Chapter 7 is theory and a scaling analysis

## Before submitting (student)

- [ ] Push the repository and check that the link in the report opens
- [ ] Read the report once, sign the declaration page
- [ ] Compare the CO1 to CO6 wording in Table 1.1 with the official course handout
- [ ] Practise with `VIVA_NOTES.md`

## Push commands

Create an empty public repository named `plant-disease-detection-dl` at https://github.com/new (no README), then:

```
git remote add origin https://github.com/Arpita-2755/plant-disease-detection-dl.git
git branch -M main
git push -u origin main
```
