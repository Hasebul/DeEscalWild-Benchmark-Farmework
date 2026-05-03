# De-escalation Video Speaker Diarization

A clean, modular pipeline that fetches video metadata and transcripts from a list of YouTube channels.

---

## Project Structure

```
youtube_scraper/
├── config.py              # All constants and settings (single source of truth)
├── logger.py              # Shared logging setup
├── channel_fetcher.py     # Stage 1 – video metadata per channel
├── csv_merger.py          # Stage 2 – merge per-channel CSVs
├── transcript_fetcher.py  # Stage 3 – enrich with transcripts
├── main.py                # CLI entry point
├── requirements.txt
│
├── Input/
│   └── Source of de-escalation youtube video playlist.csv   ← your input
│
├── output-video-list/     ← created automatically (per-channel CSVs)
└── merge-output/          ← created automatically (merged + transcript CSVs)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline
```bash
python main.py
```

### Run individual stages
```bash
    python main.py                     # process all unprocessed videos
    python main.py --model gemini-2.5-pro
    python main.py --dry-run           # validate manifest, print plan, do nothing
    python main.py --video-id abc123   # process a single video by ID
```

### Resume transcript fetching after a crash
```bash
python main.py --stage transcribe --resume 142
```

---

## Input CSV format

The file at `Input/Source of de-escalation youtube video playlist.csv` must have a `link` column containing YouTube channel URLs in the form:

```
https://www.youtube.com/@ChannelHandle
```

---

## Output files

| File | Description |
|---|---|
| `output-video-list/<channel>_metadata_and_transcripts.csv` | Raw per-channel data |
| `merge-output/video-list.csv` | All channels merged, with global `id` |
| `merge-output/video-list-with-transcripts.csv` | Final enriched file |

---

## Configuration

All tunable parameters live in **`config.py`** — paths, delays, language preferences, description length, yt-dlp options. No need to touch any other file for routine changes.
