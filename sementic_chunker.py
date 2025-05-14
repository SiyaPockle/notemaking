import os
import re
import time
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from process_pdf import extract_text_from_pdf

# Existing chunker imports and classes
import sys
# import pysqlite3
import sqlite3
sys.modules["sqlite3"] = sqlite3
from chromadb.utils import embedding_functions
from chunking_evaluation.chunking.base_chunker import BaseChunker
from chunking_evaluation import openai_token_count
from chunking_evaluation.chunking.recursive_token_chunker import RecursiveTokenChunker
import backoff

load_dotenv()

# Configuration
SUBJECTS = {
    # "Computer_Forensic_Cyber_Security": [
    #     "/workspaces/notemakinggg/Books/CFCS/Computer Forenscis John R. Vacca.pdf",
    #     "/workspaces/notemakinggg/Books/CFCS/Hacking for Dummies.pdf"
    # ]
    #,
    # "Database_Management_System": [
    #     r"Books\DBMS\Fundamental_of_Database_Systems.pdf",
    #     r"Books\DBMS\Navathe textbook.pdf",
    #     r"Books\DBMS\Ramakrishnan gehrke textbook.pdf"
    # ]
    # "Cryptography_and_Network_Security" : ["Books\CNS\Cryptography and Network Security Principles and Practices, 4th Ed - William Stallings.pdf",
    #                                        "Books\CNS\cryptography-and-network-security-forouzan-copy.pdf"],
    # "Multimedia_Systems_and_Applications" : ["Books\Multimedia\gfx-multimedia-making-it-work-8th-edition.pdf"] ,
    "Mobile_Computing" : [r"Books\MCOMP\tutorial_Mobile-Communications-JochenSchiller.pdf"]

}
CHUNK_OUTPUT_ROOT = "./subject_chunks"
CHECKPOINT_ROOT = "./chunker_checkpoints"

class GeminiRateLimitedClient:
    # (Keep the exact GeminiRateLimitedClient implementation from previous code)
    # [Include full GeminiRateLimitedClient class here]
    def __init__(self, model_name, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model_name = model_name
        
        # Rate limit tracking
        self.request_timestamps = []  # Track timestamps of requests for RPM calculation
        self.daily_request_count = 0  # Track daily requests
        self.day_start_timestamp = time.time()  # Track when the day started for RPD
        
        # Rate limits
        self.rpm_limit = 15  # 15 requests per minute
        self.rpd_limit = 1500  # 1500 requests per day
        
        # Request cache to avoid duplicate requests
        self.response_cache = {}
        
    def _wait_for_rate_limit(self):
        """Enforces rate limits by waiting if necessary"""
        current_time = time.time()
        
        # Check and reset daily counter if needed
        if current_time - self.day_start_timestamp > 86400:  # 24 hours in seconds
            self.daily_request_count = 0
            self.day_start_timestamp = current_time
            print("Daily request counter reset")
        
        # Check if we've hit daily limit
        if self.daily_request_count >= self.rpd_limit:
            wait_seconds = 86400 - (current_time - self.day_start_timestamp)
            print(f"Daily rate limit reached. Waiting {wait_seconds/3600:.2f} hours until reset")
            time.sleep(wait_seconds + 1)  # Wait until next day plus 1 second buffer
            self.daily_request_count = 0
            self.day_start_timestamp = time.time()
        
        # Clean up old timestamps (older than 60 seconds)
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        
        # Check if we're approaching RPM limit
        if len(self.request_timestamps) >= self.rpm_limit:
            # Calculate wait time (time until oldest request is 60 seconds old)
            oldest_timestamp = min(self.request_timestamps)
            wait_seconds = 60 - (current_time - oldest_timestamp)
            
            if wait_seconds > 0:
                print(f"Approaching rate limit. Waiting {wait_seconds:.2f} seconds")
                time.sleep(wait_seconds + 0.1)  # Add a small buffer
    
    def _cache_key(self, system_prompt, messages, max_tokens, temperature):
        """Create a cache key from request parameters"""
        message_str = json.dumps([msg["content"] for msg in messages])
        return f"{system_prompt}_{message_str}_{max_tokens}_{temperature}"
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=300)
    def create_message(self, system_prompt, messages, max_tokens=8192, temperature=0.2):
        # Check cache first
        cache_key = self._cache_key(system_prompt, messages, max_tokens, temperature)
        if cache_key in self.response_cache:
            print("Using cached response")
            return self.response_cache[cache_key]
            
        # Wait if needed to respect rate limits
        self._wait_for_rate_limit()
        
        try:
            print("\n--- Gemini API Request ---")
            print("Model:", self.model_name)
            print("System Prompt length:", len(system_prompt))
            print("Messages count:", len(messages))
            print("Max Tokens:", max_tokens)
            print("Temperature:", temperature)

            gpt_messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Record this request timestamp
            self.request_timestamps.append(time.time())
            self.daily_request_count += 1
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                n=1,
                messages=gpt_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            response_content = response.choices[0].message.content
            print(f"\n--- Gemini API Response (first 100 chars) ---\n{response_content[:100]}...")
            
            # Cache the response
            self.response_cache[cache_key] = response_content
            
            return response_content
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"Rate limit error: {e}, backing off and retrying...")
                # Force a longer wait on rate limit errors
                time.sleep(10)  
            else:
                print(f"Error in GeminiClient.create_message: {e}, retrying...")
            raise e

class LLMSemanticChunker(BaseChunker):
    # (Keep the exact LLMSemanticChunker implementation from previous code)
    # [Include full LLMSemanticChunker class here]
    """
    A chunker that splits text into thematically consistent chunks using the Gemini 2.0 Flash model.
    Enhanced with rate limiting and checkpoint saving for long-running processes.
    """
    def __init__(self, llm_provider="gemini", api_key=None, model_name=None, checkpoint_dir="chunker_checkpoints"):
        if llm_provider != "gemini":
            raise ValueError("Invalid llm_provider. Please use 'gemini'.")
        if model_name is None:
            model_name = "gemini-2.0-flash"
            
        self.client = GeminiRateLimitedClient(model_name, api_key=api_key)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set pre-chunk size and overlap
        self.splitter = RecursiveTokenChunker(
            chunk_size=1000,  # Increased to 1000 tokens per pre-chunk
            chunk_overlap=50,  # 50-token overlap for continuity
            length_function=openai_token_count
        )
        self.chunk_overlap = 0  # Store chunk_overlap for later use

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        """Generate the prompt for the LLM to identify split points."""
        system_content = (
            "You are an expert in textbook structure and content organization. "
            "Your task is to analyze a textbook excerpt divided into smaller chunks and identify where one complete topic or subtopic ends and another begins. "
            "The excerpt is pre-segmented into smaller chunks, marked with <|start_chunk_X|> and <|end_chunk_X|> tags (X is the chunk number). "
            "Your goal is to suggest splits after certain chunks to create final chunks that each contain a complete topic, subtopic, or key idea, suitable for detailed note-taking and in-depth understanding. "
            "Look for natural breaks in the content, such as changes in subject matter, new section headers, or transitions between concepts. "
            "Pay attention to any section headers, titles, or other structural markers within the text, as they often indicate the start of a new topic or subtopic. "
            "Output a list of chunk numbers AFTER which a split should occur. "
            "**Minimize the number of splits**. Only suggest splits at very clear and significant topic boundaries. " 
            "Aim for final chunks that are **substantial and cover a complete sub-section or topic comprehensively**, rather than very short, granular chunks." 
            "For example, if chunks 1 and 2 cover the same subtopic and chunk 3 starts a new one, suggest a split after chunk 2. "
            "Ensure the split numbers are in ascending order and that at least one split is suggested. "
            "Format your response as: 'split_after: chunk_number1, chunk_number2, ...'"
        )

        user_content = (
            f"Textbook Excerpt (Pre-Chunked):\n{chunked_input}\n\n"
            f"Identify the chunk numbers after which to split for complete thematic coherence, focusing on creating self-contained chunks suitable for detailed note-taking. "
            f"Splits must be in ascending order and numbers must be equal to or greater than {current_chunk}."
        )

        if invalid_response:
            user_content += f"\n\nPrevious invalid response: '{invalid_response}'. Provide a new valid set of split chunk numbers."

        messages = [
            {"role": "user", "content": user_content}
        ]
        return system_content, messages

    def _save_checkpoint(self, chunks, split_indices, current_chunk):
        """Save progress to a checkpoint file"""
        checkpoint_path = os.path.join(self.checkpoint_dir, "chunker_checkpoint.json")
        data = {
            "timestamp": time.time(),
            "current_chunk": current_chunk,
            "split_indices": split_indices,
            # Don't save full chunks to keep file size manageable
            "num_chunks": len(chunks)
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f)
        print(f"Checkpoint saved at chunk {current_chunk}")

    def _load_checkpoint(self):
        """Load progress from checkpoint if available"""
        checkpoint_path = os.path.join(self.checkpoint_dir, "chunker_checkpoint.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            print(f"Checkpoint loaded. Resuming from chunk {data['current_chunk']}")
            return data['current_chunk'], data['split_indices']
        return None, None

    def split_text(self, text):
        """Split the input text into semantic chunks using Gemini 2.0 Flash with rate limiting."""
        print("Starting semantic chunking with rate limiting...")
        
        # Pre-split into chunks
        chunks = self.splitter.split_text(text)
        print(f"\nInitial token-based chunks generated: {len(chunks)}")
        
        # Try to load checkpoint
        loaded_current_chunk, loaded_split_indices = self._load_checkpoint()
        
        split_indices = loaded_split_indices or []
        current_chunk = loaded_current_chunk or 0

        with tqdm(total=len(chunks), desc="Processing chunks", initial=current_chunk) as pbar:
            while current_chunk < len(chunks):
                token_count = 0
                chunked_input = ''
                input_chunk_indices = []

                # Batch pre-chunks up to ~4,000 tokens (reduced from 10,000 to stay well within limits)
                for i in range(current_chunk, len(chunks)):
                    chunk_tokens = openai_token_count(chunks[i])
                    if token_count + chunk_tokens > 4000:
                        break
                    token_count += chunk_tokens
                    chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    input_chunk_indices.append(i + 1)

                if not input_chunk_indices:
                    break

                print(f"\n--- Processing chunks {input_chunk_indices[0]} to {input_chunk_indices[-1]} ---")
                print(f"Token count for this batch: {token_count}")

                # Get split points from LLM
                system_prompt, messages = self.get_prompt(chunked_input, current_chunk)
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Add rate-limited forced delay between API calls
                        if retry_count > 0:
                            time.sleep(5)  # Extra delay for retries
                            
                        result_string = self.client.create_message(
                            system_prompt,
                            messages,
                            max_tokens=1024,  # Reduced from 8192 to minimize token usage
                            temperature=0.2
                        )

                        split_after_lines = [line for line in result_string.split('\n') if 'split_after:' in line]
                        if not split_after_lines:
                            print(f"Warning: 'split_after:' not found in LLM response.")
                            numbers = []
                            # If we've tried multiple times and still can't get it, just move on
                            if retry_count == max_retries - 1:
                                print("Maximum retries reached, continuing without splits for this section")
                                break
                            retry_count += 1
                            continue

                        numbers = list(map(int, re.findall(r'\d+', split_after_lines[0])))
                        print("Gemini suggested split numbers:", numbers)

                        if numbers and numbers == sorted(numbers) and all(num >= current_chunk for num in numbers):
                            break
                        
                        # Invalid response, retry with previous response in prompt
                        system_prompt, messages = self.get_prompt(chunked_input, current_chunk, numbers)
                        print("Invalid response format, retrying")
                        retry_count += 1
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        retry_count += 1
                        time.sleep(10)  # Longer wait after errors
                        
                # Process the results
                if numbers:
                    valid_numbers = [num for num in numbers if num > current_chunk and num <= input_chunk_indices[-1]]
                    if valid_numbers:
                        split_indices.extend(valid_numbers)
                        print("Valid split indices extended:", valid_numbers)
                    else:
                        print("Warning: No valid split indices returned by Gemini in current range.")
                else:
                    print("No split numbers found in valid format from Gemini.")
                
                # Update progress and save checkpoint
                old_current_chunk = current_chunk
                current_chunk = input_chunk_indices[-1]
                pbar.update(current_chunk - old_current_chunk)
                self._save_checkpoint(chunks, split_indices, current_chunk)
                
                # Add a delay between batches to help with rate limiting
                time.sleep(4)

        # Assemble final chunks, handling overlaps
        chunks_to_split_after = [i - 1 for i in split_indices]
        print("\nFinal split indices (0-based):", chunks_to_split_after)
        docs = []
        current_chunk = ''
        last_end_pos = 0

        print("\n--- Assembling Final Chunks ---")
        for i, chunk in enumerate(chunks):
            # Trim overlap from the start of subsequent chunks
            if i > 0 and last_end_pos > 0:
                overlap_size = self.chunk_overlap
                chunk_tokens = chunk.split()
                if len(chunk_tokens) > overlap_size:
                    chunk = ' '.join(chunk_tokens[overlap_size:])
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                print(f"Chunk {len(docs)} (ends at initial chunk {i+1}): '{current_chunk[:50]}...'")
                current_chunk = ''
                last_end_pos = i + 1
        if current_chunk:
            docs.append(current_chunk.strip())
            print(f"Last Chunk {len(docs)} (remaining): '{current_chunk[:50]}...'")

        print(f"\nFinal number of semantic chunks generated: {len(docs)}")
        
        # Save final chunks to checkpoint directory
        final_chunks_path = os.path.join(self.checkpoint_dir, "final_chunks.json")
        with open(final_chunks_path, 'w') as f:
            json.dump({"count": len(docs), "timestamp": time.time()}, f)
            
        return docs

# def process_subject(subject_name, textbook_paths):
#     """Process all textbooks for a subject and save combined chunks"""
#     subject_chunk_dir = os.path.join(CHUNK_OUTPUT_ROOT, subject_name)
#     os.makedirs(subject_chunk_dir, exist_ok=True)
    
#     combined_chunks = []
#     failed_textbooks = []
    
#     for textbook_path in textbook_paths:
#         if not os.path.exists(textbook_path):
#             print(f"File not found: {textbook_path}")
#             failed_textbooks.append(textbook_path)
#             continue
            
#         textbook_name = os.path.basename(textbook_path).rsplit('.', 1)[0]
#         checkpoint_dir = os.path.join(CHECKPOINT_ROOT, subject_name, textbook_name)
        
#         try:
#             # Extract text from PDF
#             print(f"\nProcessing {textbook_name}...")
#             start_time = time.time()
#             textbook_content = extract_text_from_pdf(textbook_path)
#             extract_time = time.time() - start_time
#             print(f"Text extraction completed in {extract_time:.2f}s")
            
#             # Initialize chunker with textbook-specific checkpoint
#             chunker = LLMSemanticChunker(
#                 api_key=os.environ.get("GEMINI_API_KEY"),
#                 checkpoint_dir=checkpoint_dir
#             )
            
#             # Generate semantic chunks
#             start_chunking = time.time()
#             textbook_chunks = chunker.split_text(textbook_content)
#             chunking_time = time.time() - start_chunking
#             print(f"Chunking completed in {chunking_time:.2f}s")
            
#             # combined_chunks.extend(textbook_chunks)
#             print(f"Added {len(textbook_chunks)} chunks from {textbook_name}")
            
#             # Save individual textbook chunks
#             textbook_chunk_path = os.path.join(subject_chunk_dir, f"{textbook_name}_chunks.txt")
#             with open(textbook_chunk_path, "w", encoding="utf-8") as f:
#                 f.write("\n\n".join(textbook_chunks))
            
#         except Exception as e:
#             print(f"Error processing {textbook_name}: {str(e)}")
#             failed_textbooks.append(textbook_path)
#             continue

#     # Save combined chunks for subject
#     if combined_chunks:
#         output_path = os.path.join(subject_chunk_dir, "combined_chunks.txt")
#         with open(output_path, "w") as f:
#             f.write("\n\n".join(combined_chunks))
#         print(f"\nSaved {len(combined_chunks)} combined chunks for {subject_name}")
    
#     return failed_textbooks

def process_subject(subject_name, textbook_paths):
    """Process all textbooks for a subject and save combined chunks"""
    subject_chunk_dir = os.path.join(CHUNK_OUTPUT_ROOT, subject_name)
    os.makedirs(subject_chunk_dir, exist_ok=True)
    
    combined_chunks = []
    failed_textbooks = []
    
    for textbook_path in textbook_paths:
        if not os.path.exists(textbook_path):
            print(f"File not found: {textbook_path}")
            failed_textbooks.append(textbook_path)
            continue
            
        textbook_name = os.path.basename(textbook_path).rsplit('.', 1)[0]
        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, subject_name, textbook_name)
        
        try:
            # Extract text from PDF
            print(f"\nProcessing {textbook_name}...")
            start_time = time.time()
            textbook_content = extract_text_from_pdf(textbook_path)
            extract_time = time.time() - start_time
            print(f"Text extraction completed in {extract_time:.2f}s")
            
            # Initialize chunker with textbook-specific checkpoint
            chunker = LLMSemanticChunker(
                api_key=os.environ.get("GEMINI_API_KEY"),
                checkpoint_dir=checkpoint_dir
            )
            
            # Generate semantic chunks
            start_chunking = time.time()
            textbook_chunks = chunker.split_text(textbook_content)
            chunking_time = time.time() - start_chunking
            print(f"Chunking completed in {chunking_time:.2f}s")
            
            # Add to combined list
            combined_chunks.extend(textbook_chunks)
            print(f"Added {len(textbook_chunks)} chunks from {textbook_name}")
            
            # Save individual textbook chunks
            textbook_chunk_path = os.path.join(subject_chunk_dir, f"{textbook_name}_chunks.txt")
            with open(textbook_chunk_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(textbook_chunks))
            
        except Exception as e:
            print(f"Error processing {textbook_name}: {str(e)}")
            failed_textbooks.append(textbook_path)
            continue

    # Save combined chunks for subject
    if combined_chunks:
        output_path = os.path.join(subject_chunk_dir, "combined_chunks.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_chunks))
        print(f"\nSaved {len(combined_chunks)} combined chunks for {subject_name}")
    
    return failed_textbooks


def main():
    parser = argparse.ArgumentParser(description="Semantic Chunking Pipeline")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume previous processing")
    args = parser.parse_args()
    
    if args.resume:
        print("\nResuming previous processing...")
    
    total_failed = []
    for subject, paths in SUBJECTS.items():
        print(f"\n{'='*40}")
        print(f"Processing Subject: {subject.replace('_', ' ')}")
        print(f"{'='*40}")
        
        failed = process_subject(subject, paths)
        total_failed.extend(failed)
        
        if failed:
            print(f"\nFailed textbooks for {subject}:")
            for path in failed:
                print(f"- {path}")
    
    if total_failed:
        print("\nProcessing completed with errors in following textbooks:")
        for path in total_failed:
            print(f"- {path}")
    else:
        print("\nAll textbooks processed successfully!")

if __name__ == "__main__":
    main()