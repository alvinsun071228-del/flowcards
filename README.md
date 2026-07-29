# FlowCards — AI-Powered Flashcard Study App

FlowCards is a full-stack flashcard application that generates study decks from
topics, pasted notes, or uploaded files (TXT, Markdown, CSV, JSON, PDF, DOCX).
DeepSeek V4 Pro generates summaries, knowledge points, imagery, and
question-and-answer cards. Decks and study progress persist in browser
localStorage.

## Features

### Study System
- **6 views:** Landing, Home, AI Studio, Flow (scroll study), Quiz, Profile
- **Spaced repetition:** Full SM-2 algorithm (SuperMemo 2) with ease factor,
  progressive intervals, and adaptive scheduling
- **Quiz mode:** Multiple choice with word→definition, definition→word, and mixed
  directions
- **Dark mode:** Full theme system with animated transitions, respects
  `prefers-color-scheme` and persists to localStorage
- **Keyboard navigation:** Arrows, Enter, 1/2/3 for rating, wheel for next card
- **Undo ratings:** 5-second undo window after rating a card

### AI Generation
- Topic-to-flashcard generation
- File and notes summarization (PDF/DOCX browser-side extraction via pdf.js + mammoth.js)
- Knowledge-point extraction
- Auto-estimate deck size with heuristic fallback
- **Wikimedia Commons image search** — attach relevant, attributed images to each card

### Data Management
- Manual deck and flashcard creation
- **Card editing** — modify any card's word, translation, part of speech, or example
- **JSON import/export** — backup all decks, cards, and progress to a single file
- **Cross-deck search** — search all cards by word or definition
- Custom deck creation with emoji and description

### Technical Quality
- **Responsive:** 14 breakpoints, mobile-first layout
- **Accessible:** 59 ARIA attributes, keyboard navigation, `prefers-reduced-motion`
- **PWA:** Service worker for offline caching, manifest for "Add to Home Screen"
- **Rate limiting:** 10 req/min per IP on the AI endpoint
- **Model fallback:** DeepSeek V4 Pro → V4 Flash on rate limit
- **Error handling:** Input validation, timeout handling, user-facing error messages

## Local Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your private environment file:
   ```bash
   cp .env.example .env
   ```

4. Open `.env` and replace the placeholder with a DeepSeek API key. Never commit `.env`.

5. Start the application:
   ```bash
   flask --app app run --debug
   ```

6. Open <http://127.0.0.1:5000>.

## API Contract

### `POST /api/generate`

Request:
```json
{
  "mode": "topic",
  "prompt": "Create a beginner deck about photosynthesis.",
  "sourceText": "",
  "deckName": "Photosynthesis",
  "count": 8,
  "includeImages": true
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `mode` | string | yes | `"topic"`, `"file"`, or `"notes"` |
| `prompt` | string | for topic | Topic description (min 8 chars) |
| `sourceText` | string | for file/notes | Source material (min 80 chars) |
| `deckName` | string | no | Defaults to "AI Study Deck" |
| `count` | integer | no | 1–200, defaults to 20 |
| `includeImages` | boolean | no | Searches Wikimedia Commons for card images |

Response:
```json
{
  "deckName": "Photosynthesis",
  "summary": "Photosynthesis converts light energy into chemical energy in plants.",
  "knowledgePoints": ["Chlorophyll absorbs light", "Produces glucose and oxygen"],
  "cards": [
    {
      "question": "What is the overall equation for photosynthesis?",
      "answer": "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂",
      "imageQuery": "photosynthesis diagram labeled",
      "image": {
        "url": "https://upload.wikimedia.org/...",
        "sourceUrl": "https://commons.wikimedia.org/wiki/File:...",
        "alt": "Photosynthesis diagram",
        "credit": "Wikimedia Commons"
      }
    }
  ]
}
```

### `GET /api/health`

```json
{"ok": true, "provider": "deepseek", "model": "deepseek-v4-pro", "configured": true}
```

### `POST /api/estimate`

Request: `{"prompt": "...", "sourceText": "..."}`

Response: `{"estimatedCount": 25, "reasoning": "One-sentence explanation"}`

## Testing

```bash
# Backend tests
python3 -m unittest discover -s tests -v

# Frontend test (requires Playwright)
pip install playwright && playwright install chromium
python3 tests/test_frontend.py
```

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS (~400KB single file) |
| Backend | Flask (Python 3) |
| AI Provider | DeepSeek API (V4 Pro / V4 Flash) |
| Storage | Browser localStorage |
| Image Search | Wikimedia Commons API |
| File Parsing | pdf.js + mammoth.js (browser-side) |
| Animations | GSAP (ScrollTrigger, timelines) |
| Icons | Lucide |
| Hosting | Vercel (serverless) |

## Known Limitations

- Browser localStorage is device-specific and does not provide user accounts.
- The in-memory rate limit resets when the server restarts and is not shared
  between multiple Gunicorn workers.
- AI-generated cards should be reviewed for accuracy before studying.
- Image search depends on Wikimedia Commons availability and may return no
  results for abstract or niche topics.
- PWA offline support caches static assets only; AI generation requires
  network connectivity.
