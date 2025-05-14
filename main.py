from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict
import os
from dotenv import load_dotenv
import time



# Load environment variables
load_dotenv()


def generate_notes(subject: str, queries: List[str], k: int = 8) -> Dict[str, str]:
    """
    Generate notes for multiple queries, handling each query's context separately.
    Returns a dictionary with questions as keys and answers as values.
    Also prints and saves top-k chunks per query.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Gemini embedding model
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    
    # Load vector store once per subject
    vector_store = FAISS.load_local(
        f"vector_stores/{subject}", 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    prompt_template = """
You are an expert academic assistant in the subject of {subject}, generating highly detailed, concept-oriented study notes. Use the following context to answer the user query with depth and accuracy.

Subject: {subject}

Context:
{context}

Task:
Generate exhaustive, expert-level study notes explaining and elaborating on the following topic:
❖ {query}

Instructions:

Your primary goal is to teach the concept clearly and thoroughly, just like a top-tier textbook or university professor would.

Follow a natural progression of subtopics based on the academic structure used in textbooks or university courses.

Explain each subtopic clearly: if the concept requires deeper insight or formulas, expand it accordingly.

Do not use a rigid predefined template. Instead, create a structure that flows logically and completely explains the topic.

Use headings and subheadings to break down the topic naturally: include only what is applicable and necessary.

Where appropriate, include:

Definitions and formal explanations

Diagrams or visuals in mermaid to explain processes or relationships

Examples of varying complexity

Mathematical derivations, formulas, or pseudocode

Real-world use cases and scenarios

Historical background (only if essential to understand the evolution of the topic)

Connections to broader topics or prerequisite knowledge

Troubleshooting or clarifying commonly confused parts

Guidelines:

Use **bold** for important terms and *italics* for emphasis.

Include 5–7 varied examples with specific scenarios to make the concept intuitive.

Aim for mastery-level notes, suitable for final-year undergraduate or early post-grad level.

The tone should be academically rigorous but approachable, like an elite mentor teaching a dedicated student.

Ensure that each point builds towards understanding the topic deeply, rather than just listing facts.

Include visual learning aids (2+ mermaid diagrams or flowcharts).

Checklist before finalizing: ✅ Subtopics naturally divided and elaborated
✅ No unnecessary sections; everything adds clarity
✅ Rich, theory-driven explanations with clarity
✅ Specific examples from different domains
✅ Visuals and diagrams for hard-to-grasp relationships
✅ Deep conceptual coverage of the topic

Now create the notes:
"""

    
    prompt = PromptTemplate(
        input_variables=["subject", "context", "query"],
        template=prompt_template
    )
    
    note_chain = LLMChain(llm=llm, prompt=prompt)
    
    results = {}


    # with open("top_k_chunks.txt", "a", encoding="utf-8") as chunk_file:
    for query in queries:
        # Get context for each query separately
        docs = vector_store.similarity_search(query, k=k)

    

        # Create limited-size context for prompt
        context = "\n---\n".join([doc.page_content[:3000] for doc in docs])[:80000]
        
        # Generate notes
        response = note_chain.invoke({"subject": subject, "context": context, "query": query})
        results[query] = response['text']
    
    # Create a directory for storing retrieved chunks
    # chunk_dir = "retrieved_chunks"
    # os.makedirs(chunk_dir, exist_ok=True)
    
    # for idx, query in enumerate(queries, 1):
    #     # Get context for each query separately
    #     docs = vector_store.similarity_search(query, k=k)

    #     # Save each retrieved chunk separately
    #     query_dir = os.path.join(chunk_dir, f"query_{idx}")
    #     os.makedirs(query_dir, exist_ok=True)

    #     for i, doc in enumerate(docs):
    #         chunk_path = os.path.join(query_dir, f"chunk_{i+1}.txt")
    #         with open(chunk_path, "w", encoding="utf-8") as file:
    #             file.write(doc.page_content)

    #     # Combine chunks for LLM input
    #     context = "\n---\n".join([doc.page_content[:3000] for doc in docs])[:80000]

    #     # Generate notes for each query
    #     response = note_chain.invoke({"subject":subject ,"context": context, "query": query})
    #     results[query] = response['text']

    
    return results

def format_notes(notes_dict: Dict[str, str]) -> str:
    """Format multiple notes into a structured document."""
    formatted = "# Comprehensive Study Notes\n\n"
    for idx, (question, answer) in enumerate(notes_dict.items(), 1):
        formatted += f"## Topic {idx}: {question}\n\n"
        formatted += f"{answer}\n\n"
        formatted += "---\n\n"
    return formatted

def extract_questions_with_llm(text: str) -> List[str]:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    extraction_prompt = """Extract all the questions or topics from the following text. Understand the question meaning based on the semantics of the text and If it is a new question separate it with a delimiter like | and INCLUDE THE WHOLE QUESTION AS IT IS DO NOT CHANGE THE QUESTION ONLY ADD | AT THE END OF EACH:

Text: {text}

Questions/Topics:
- """
    
    prompt = PromptTemplate( 
        input_variables=["text"],
        template=extraction_prompt
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.invoke({"text": text})
    questions = [line.strip("| ").strip() for line in response['text'].split("\n") if line.strip()]
    return questions

def get_user_input():
    print("Welcome to the Dynamic Study Notes Generator!")
    subject = input("Enter the subject (e.g., 'Database_Management_System'): ").strip()
    
    print("\nEnter all your queries at once. The system will automatically detect individual questions/topics:")
    queries_input = input("\nEnter queries: ").strip()
    
    queries = extract_questions_with_llm(queries_input)
    
    return subject, queries




if __name__ == "__main__":
    subject, queries = get_user_input()
    
    start_time = time.time()  # Start timer
    
    notes_dict = generate_notes(subject=subject, queries=queries)
    
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    
    formatted_notes = format_notes(notes_dict)
    print(formatted_notes)
    
    with open("notes.md", "w", encoding="utf-8") as file:
        file.write(formatted_notes)
    
    print(f"✅ Notes generated in {elapsed_time:.2f} seconds.")


