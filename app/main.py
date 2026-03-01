import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types

# Connecting to Vertex AI with GCP Project ID.
GCP_PROJECT = os.getenv("GCP_PROJECT", "hong-agentic-ai-p1")
GCP_LOCATION = "us-central1"
client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Setting up a FastAPI web server, and Jinja2Templates for finding HTML.
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Loading the "NY Times Top 50 movies from 21st Century" dataset from movies.json
MOVIES_PATH = Path("movies.json")
with open(MOVIES_PATH) as f:
    MOVIES = json.load(f)

# Converting movie data into readable text covering key info.
def build_movie_list_text(movies):
    lines = []
    for m in movies:
        genres = ", ".join(m["genres"])
        themes = ", ".join(m["themes"])
        mood   = ", ".join(m["mood"])
        lines.append(
            f"{m['rank']}. {m['title']} ({m['year']}) — Dir: {m['director']} | "
            f"Genres: {genres} | Themes: {themes} | Mood: {mood}\n"
            f"   {m['description']}"
        )
    return "\n\n".join(lines)
MOVIE_LIST_TEXT = build_movie_list_text(MOVIES)

# Load the system prompt from prompt.txt with the full movie list as {MOVIE_LIST}
PROMPT_PATH = Path("app/prompt.txt")
with open(PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read().replace("{MOVIE_LIST}", MOVIE_LIST_TEXT)

# Print the length of the system prompt for verification.
print(f"System prompt loaded: {len(SYSTEM_PROMPT)} characters")

# Safety filters and python backstops.
DISTRESS_PATTERNS = re.compile(
    r"\b(suicide|kill myself|end my life|self.harm|want to die|hurt myself)\b",
    re.IGNORECASE,
)

VIOLENCE_PATTERNS = re.compile(
    r"\b(kill others|hurt someone|harm others|attack someone)\b",
    re.IGNORECASE,
)

OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(recipe|cooking|stock market|cryptocurrency|bitcoin|weather forecast"
    r"|sports score|book flight|hotel booking|news headlines|political party"
    r"|medication dosage|math homework|write code|computer program)\b",
    re.IGNORECASE,
)

DISTRESS_RESPONSE = (
    "I've noticed something in your message that concerns me. If you're going through a difficult time, please reach out to the 988 Suicide & Crisis Lifeline. "
)

VIOLENCE_RESPONSE = (
    "That topic is outside what I can help with. If you or someone you know is in danger, please call 911. Do you want me to recommend a film instead?"
)

OUT_OF_SCOPE_RESPONSE = (
    "That topic is outside what I can help with. I'm Dr. Cinema, specialized in recommending films from the NYT Top 50 greatest movies of the 21st century. "
    "Do you want me to suggest something based on a mood, theme, or genre instead?"
)

# Checking the user's message against all three safety filters.
def backstop_check(user_message: str):
    if DISTRESS_PATTERNS.search(user_message):
        return DISTRESS_RESPONSE
    if OUT_OF_SCOPE_PATTERNS.search(user_message):
        return OUT_OF_SCOPE_RESPONSE
    return None

# Setting up homepage (localhost:8000) with index file.
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Setting up chat endpoints to handle message requests. 
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "").strip()
    history = body.get("history", [])

    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Running backstop safety check before calling Vertex AI.
    forced = backstop_check(user_message)
    if forced:
        return JSONResponse({"reply": forced})

    # Building full conversation history for Vertex AI. 
    contents = []

    # Injecting system prompt as the first message to ensure context.
    contents.append(types.Content(role="user", parts=[types.Part(text=SYSTEM_PROMPT)]))
    contents.append(types.Content(role="model", parts=[types.Part(text="Understood! I am Dr. Cinema, ready to recommend films exclusively from the NYT Top 50 greatest films of the 21st century.")]))

    #  Adding all messages from the history.
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    # Sending everything to Vertex AI and receiving response, with set parameters.
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=600,
            ),
        )
        reply = response.text

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"reply": reply})
