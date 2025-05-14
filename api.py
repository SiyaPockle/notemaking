# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from main import extract_questions_with_llm, generate_notes, format_notes
from md_convert import convert_md_to_html_string  # new import

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteRequest(BaseModel):
    subject: str
    raw_queries: List[str]

@app.post("/generate-notes")
async def generate_notes_endpoint(req: NoteRequest):
    queries = extract_questions_with_llm(req.raw_queries)
    notes_md = format_notes(generate_notes(req.subject, queries))
    notes_html = convert_md_to_html_string(notes_md)
    print(notes_html)
    return {
        "queries": queries,
        "notes_html": notes_html
    }
