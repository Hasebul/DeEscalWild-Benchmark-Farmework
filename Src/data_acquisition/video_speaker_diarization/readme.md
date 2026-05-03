# YouTube Speaker Diarization Pipeline

A clean, modular pipeline that downloads YouTube audio, sends it to Google Gemini, and returns structured speaker-diarized transcripts saved as both TXT and CSV.

---

## Project Structure

```
diarization_pipeline/
│
├── config.py           # All paths, model names, and tunable settings
├── logger.py           # Shared timestamped logger
├── auth.py             # Gemini client authentication (env vars only — no hardcoded keys)
├── models.py           # Pydantic schemas for structured Gemini responses
├── downloader.py       # Downloads YouTube audio as MP3 via yt-dlp
├── transcriber.py      # Sends audio to Gemini; parses diarized transcription
├── exporter.py         # Saves TXT, CSV outputs and appends to the run log
├── status_tracker.py   # JSON-based resume state (processed / ignored IDs)
├── main.py             # CLI entry point — orchestrates the full pipeline
│
├── data/
│   └── youtube_video_manifest.csv    ← your input file
│
├── temp_downloads/                   ← created automatically (deleted after each run)
│
└── output/
    ├── txt/            ← human-readable dialogue logs
    ├── csv/            ← structured transcript rows
    ├── log.csv         ← per-video token usage and timing
    └── process_status.json  ← resume checkpoint
```

---

## Installation

```bash
pip install google-genai yt-dlp pydantic tenacity tqdm pandas
```

> **FFmpeg is required** for audio conversion. Install it via your OS package manager:
> ```bash
> # macOS
> brew install ffmpeg
>
> # Ubuntu / Debian
> sudo apt install ffmpeg
>
> # Windows (via Chocolatey)
> choco install ffmpeg
> ```

---

## Authentication

**No API keys are ever hardcoded.** Set credentials via environment variables before running.

### Option A — Google AI Studio (recommended for personal use)

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### Option B — Vertex AI (for Google Cloud projects)

```bash
export GOOGLE_GENAI_USE_VERTEXAI="true"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"   # optional, defaults to "global"
```

---

## Input Format

The manifest CSV at `data/youtube_video_manifest.csv` must contain these columns:

| Column    | Description                              | Example                                    |
|-----------|------------------------------------------|--------------------------------------------|
| `id`      | Unique video identifier                  | `abc123`                                   |
| `URL`     | Full YouTube watch URL                   | `https://www.youtube.com/watch?v=abc123`   |
| `minutes` | Video duration in minutes (for filtering)| `12.5`                                     |

---

## Usage

### Run the full pipeline on all unprocessed videos

```bash
python main.py
```

### Choose a specific Gemini model

```bash
python main.py --model gemini-2.5-pro
python main.py --model gemini-2.5-flash    # default
python main.py --model gemini-2.0-flash-exp
```

### Process a single video by ID

```bash
python main.py --video-id abc123
```

### Dry run — validate manifest and preview what would be processed

```bash
python main.py --dry-run
```

### Resume after a crash

The pipeline checkpoints automatically after every video. Just re-run the same command — already-processed videos are skipped:

```bash
python main.py
```

---

## Output Files

### `output/txt/{id}.txt` — Human-readable dialogue log

```
--- Detailed Conversation Log ---

[00:03] Officer Johnson: Sir, can I see your ID please?
[00:07] [Civilian]: What's this about?
[00:10] Officer Johnson: Routine traffic stop. License and registration.
```

### `output/csv/{id}.csv` — Structured transcript rows

```
start_time,speaker_label,transcript
00:03,Officer Johnson,Sir can I see your ID please?
00:07,[Civilian],What's this about?
00:10,Officer Johnson,Routine traffic stop. License and registration.
```

### `output/log.csv` — Per-video processing log

```
id,input_token,output_token,thought_token,execution_time_sec,model_name
abc123,14823,612,0,18.42,gemini-2.5-flash
def456,11204,489,0,14.71,gemini-2.5-flash
```

### `output/process_status.json` — Resume checkpoint

```json
{
    "processed": ["abc123", "def456"],
    "ignore":    ["xyz999"]
}
```

Videos in `processed` were completed successfully. Videos in `ignore` were permanently skipped (e.g. exceeded the max duration limit).

---

## Configuration

All tunable parameters live in **`config.py`** — edit them there without touching any logic file.

| Setting                    | Default             | Description                                      |
|----------------------------|---------------------|--------------------------------------------------|
| `Model.DEFAULT`            | `gemini-2.5-flash`  | Gemini model used for transcription              |
| `MAX_VIDEO_DURATION_MIN`   | `45`                | Videos longer than this (minutes) are skipped    |
| `AUDIO_FORMAT`             | `mp3`               | Downloaded audio format                          |
| `AUDIO_QUALITY`            | `192`               | Audio bitrate in kbps                            |
| `TRANSCRIPT_TEMPERATURE`   | `0.0`               | LLM temperature (0 = deterministic)              |
| `RETRY_MAX_ATTEMPTS`       | `7`                 | Max API retry attempts on failure                |
| `UNKNOWN_SPEAKER_MARKER`   | `?`                 | Placeholder when speaker info cannot be found    |

---

## Module Responsibilities

| Module              | Responsibility                                                      |
|---------------------|---------------------------------------------------------------------|
| `config.py`         | Single source of truth for all constants — the only file to tune    |
| `logger.py`         | Consistent timestamped logging across all modules                   |
| `auth.py`           | Reads env vars, initializes and validates the Gemini client         |
| `models.py`         | Pydantic schemas; `VideoTranscription.label_for_voice()` helper     |
| `downloader.py`     | yt-dlp wrapper — download + convert to MP3, delete temp files       |
| `transcriber.py`    | Builds Gemini request, calls API with retry, parses JSON response   |
| `exporter.py`       | Writes TXT, CSV, and appends to the run log                         |
| `status_tracker.py` | Loads/saves `process_status.json`; provides the skip-ID set         |
| `main.py`           | CLI parser, manifest loader, per-video orchestration loop           |

---

## Error Handling

- **Download failure** — logged and skipped; video is not marked as processed so it will retry on the next run.
- **Gemini API errors** — retried up to `RETRY_MAX_ATTEMPTS` times with incremental back-off.
- **Videos over duration limit** — logged, appended to log.csv with zero tokens, and marked as `ignored` so they are never retried.
- **Keyboard interrupt** — exits cleanly; all previously completed videos remain checkpointed.
- **Any other unhandled error** — logged with full details; pipeline continues to the next video.
