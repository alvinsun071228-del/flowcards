# FlowCards with DeepSeek

FlowCards is a Flask flashcard application that can generate study decks from
a topic, pasted notes, or uploaded TXT, Markdown, CSV, JSON, PDF, and DOCX
files. DeepSeek V4 Pro generates a summary, knowledge points, and
question-and-answer cards by default. Decks and study progress persist in
browser localStorage.

## Features

- Manual deck and flashcard creation
- DeepSeek topic-to-flashcard generation
- User-entered deck sizes without fixed preset choices
- File and notes summarization
- Knowledge-point extraction
- PDF and DOCX browser-side text extraction
- Saved decks, quizzes, spaced-repetition ratings, and progress statistics
- Responsive interface with GSAP state transitions

## Local setup

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

4. Open `.env` and replace the placeholder with a newly rotated DeepSeek API
   key. Never commit `.env`.

5. Start the application:

   ```bash
   flask --app app run --debug
   ```

6. Open <http://127.0.0.1:5000>.

## Point requirements

- **Persistent data:** Decks, cards, and learning progress are stored in
  localStorage by `templates/index.html`.
- **Meaningful POST endpoint:** `POST /api/generate` in `app.py` validates the
  request, calls DeepSeek, and returns generated study content.
- **Public hosting ready:** The app includes Gunicorn and can run on an AWS
  instance with:

  ```bash
  gunicorn --bind 0.0.0.0:${PORT:-8000} app:app
  ```

Configure `DEEPSEEK_API_KEY` as a server-side environment variable in the
hosting platform. Do not place it in the HTML, repository, or deployment logs.

## Public website

`index.html` is the public GitHub Pages build. It includes the full responsive
FlowCards interface, local deck storage, study flow, quizzes, and the
browser-based demonstration generator.

The Flask application in `app.py` provides the real DeepSeek integration when
run locally or on a Python server. GitHub Pages serves static files only, so the
public Pages build cannot expose or call a private server-side API key.

## API contract

`POST /api/generate` accepts:

```json
{
  "mode": "topic",
  "prompt": "Create a beginner deck about photosynthesis.",
  "sourceText": "",
  "deckName": "Photosynthesis",
  "count": 8
}
```

`count` is entered by the learner rather than selected from fixed presets. It
must be a positive whole number. Very large requests may return fewer cards
when the source material or AI provider cannot support the requested amount.

It returns:

```json
{
  "deckName": "Photosynthesis",
  "summary": "A concise subject summary.",
  "knowledgePoints": ["Important idea"],
  "cards": [
    {
      "question": "What is photosynthesis?",
      "answer": "Photosynthesis converts light energy into chemical energy."
    }
  ]
}
```

## Known limitations

- Browser localStorage is device-specific and does not provide user accounts.
- The in-memory rate limit resets when the server restarts and is not shared
  between multiple Gunicorn workers.
- AI-generated cards should be reviewed for accuracy before studying.
