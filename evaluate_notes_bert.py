from bert_score import score
import os
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


def read_file_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return nltk.sent_tokenize(content)

def evaluate_with_bertscore(generated_chunks, reference_chunks):
    # Align lengths by repeating the reference text (or truncate)
    max_len = min(len(generated_chunks), len(reference_chunks))
    candidate = generated_chunks[:max_len]
    reference = reference_chunks[:max_len]

    # BERTScore returns precision, recall, f1
    P, R, F1 = score(candidate, reference, lang="en", verbose=True)
    
    return {
        "Average Precision": P.mean().item(),
        "Average Recall": R.mean().item(),
        "Average F1 Score": F1.mean().item()
    }

if __name__ == "__main__":
    notes_path = "notes.md"
    reference_path = "top_k_chunks.txt"

    if not os.path.exists(notes_path) or not os.path.exists(reference_path):
        print("Ensure both 'notes.md' and 'chunks_test.txt' are present.")
        exit()

    generated_notes = read_file_chunks(notes_path)
    reference_text = read_file_chunks(reference_path)

    results = evaluate_with_bertscore(generated_notes, reference_text)

    print("\n📊 BERTScore Evaluation")
    print("---------------------------")
    for metric, value in results.items():
        print(f"{metric:<20}: {value:.4f}")
    print("---------------------------")
    print("Note: BERTScore F1 focuses on semantic similarity between texts.")
