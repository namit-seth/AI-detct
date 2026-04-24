import re
import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def calculate_burstiness(text):
    """
    Calculates burstiness by finding the standard deviation of sentence lengths.
    """
    sentences = re.split(r'[.!?]+', text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
    
    if len(sentence_lengths) < 2:
        return 0.0 
        
    burstiness_score = np.std(sentence_lengths)
    return float(burstiness_score)

def calculate_perplexity(text):
    """
    Calculates perplexity using GPT-2. 
    """
    model_id = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    import transformers
    transformers.logging.set_verbosity_error()
    
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity_score = torch.exp(loss)
        
    return perplexity_score.item()

def count_ai_buzzwords(text):
    """
    Scans the text for highly overused AI vocabulary and phrases.
    """
    ai_buzzwords = [
        "delve", "tapestry", "foster", "testament", "crucial", 
        "dynamic", "multifaceted", "seamless", "underscore", 
        "pivotal", "navigate", "landscape", "beacon", "synergy",
        "intricate", "nuanced", "embark", "realm", "moreover"
    ]
    
    text_lower = text.lower()
    buzzword_count = 0
    found_words = []
    
    for word in ai_buzzwords:
        matches = re.findall(rf'\b{word}\b', text_lower)
        if matches:
            buzzword_count += len(matches)
            found_words.extend([word] * len(matches))
            
    return buzzword_count, found_words

# ==========================================
# Run the detector on a file
# ==========================================
if __name__ == "__main__":
    print("===================================")
    print("      AI Text Detector Setup       ")
    print("===================================")
    
    # 1. Ask the user for the file name
    file_path = input("Enter the name of the text file to analyze (e.g., essay.txt): ")
    
    # 2. Check if the file actually exists before trying to read it
    if not os.path.exists(file_path):
        print(f"\nError: Could not find a file named '{file_path}'.")
        print("Make sure the file is in the same folder as this script, or provide the full file path.")
    else:
        # 3. Read the file
        try:
            # We use utf-8 encoding to prevent errors from weird characters like smart quotes
            with open(file_path, 'r', encoding='utf-8') as file:
                sample_text = file.read()
                
            if not sample_text.strip():
                print("\nError: The file is empty!")
            else:
                print(f"\nReading '{file_path}'...")
                print("Loading model and analyzing text...\n")
                
                # Run the analysis
                burstiness = calculate_burstiness(sample_text)
                perplexity = calculate_perplexity(sample_text)
                buzzword_count, found_words = count_ai_buzzwords(sample_text)
                
                # Print results
                print(f"--- Structural Results ---")
                print(f"Burstiness Score: {burstiness:.2f}")
                print(f"Perplexity Score: {perplexity:.2f}")
                
                print(f"\n--- Vocabulary Results ---")
                print(f"AI Buzzwords Found: {buzzword_count}")
                if buzzword_count > 0:
                    print(f"Words: {', '.join(set(found_words))}")
                
                print("\n--- Final Verdict ---")
                if buzzword_count >= 3:
                    print("Conclusion: Highly likely to be AI-generated. (Flagged by Vocabulary Scanner)")
                elif perplexity < 60 and burstiness < 6:
                    print("Conclusion: Highly likely to be AI-generated. (Flagged by Structural Math)")
                elif perplexity > 70 and burstiness > 8 and buzzword_count == 0:
                    print("Conclusion: Highly likely to be Human-written.")
                else:
                    print("Conclusion: Mixed signals. Could be human-edited AI or a highly structured human writer.")
                    
        except Exception as e:
            print(f"\nAn error occurred while reading the file: {e}")