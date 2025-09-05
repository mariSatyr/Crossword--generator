
# CSV → Crossword (Streamlit)

A lightweight Streamlit web app that turns a CSV of vocabulary and definitions into a playable crossword layout. It also lets users add entries manually via text boxes.

## Features
- Upload CSV (`word,clue`) or paste rows in a text area
- Add words & clues manually
- Simple greedy placement with automatic numbering
- Live HTML preview of the grid
- Download puzzle JSON and a blank grid `.txt`

## Quick Start (Local)

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (Free)
1. Push this folder to a **public GitHub repo**.
2. Go to https://streamlit.io/cloud, choose **New app**.
3. Point to your repo + branch, set the file to `app.py`.
4. Click **Deploy**. Done!

> You can keep working in GitHub; commits auto-redeploy.

## CSV Format
- Two columns: `word,clue` (header optional).
- Extra columns are ignored.
- Words are normalized to letters A–Z and uppercased.

Example:

```csv
banana, A yellow fruit
python, A programming language
```

## Notes & Tips
- The placer is intentionally simple for speed and transparency; if some entries don't fit, they're listed under **Skipped**. Try a larger grid (e.g., 17–21) or a smaller/cleaner word list.
- You can download the puzzle as JSON to reuse in other apps.
- For classroom use, share the deployed Streamlit URL; students can upload their own lists or type entries directly.

## License
MIT
