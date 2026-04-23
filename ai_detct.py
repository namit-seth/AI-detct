import re
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def calculate_burstiness(text):
    """
    Calculates burstiness by finding the standard deviation of sentence lengths.
    """
    # Split the text into sentences based on punctuation (. ! ?)
    sentences = re.split(r'[.!?]+', text)
    
    # Count the number of words in each valid sentence
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
    
    # If there is only 1 sentence, there is no variance
    if len(sentence_lengths) < 2:
        return 0.0 
        
    # Burstiness is the standard deviation of these lengths
    burstiness_score = np.std(sentence_lengths)
    return float(burstiness_score)

def calculate_perplexity(text):
    """
    Calculates perplexity using GPT-2. 
    Note: This simple version works best for texts under 1024 words.
    """
    # Load the GPT-2 model and tokenizer
    model_id = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu" # Uses GPU if you have one, otherwise CPU
    
    # Suppress warning logs for cleaner output
    import transformers
    transformers.logging.set_verbosity_error()
    
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    
    # Convert text into numbers (tokens) the model can understand
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Have the model evaluate the text
    with torch.no_grad():
        # By passing the inputs as labels, the model calculates how "surprised" 
        # it is by the sequence of words (the loss).
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Perplexity is the mathematical exponential of the loss
        perplexity_score = torch.exp(loss)
        
    return perplexity_score.item()

# ==========================================
# Test the detector
# ==========================================
if __name__ == "__main__":
    # A mix of long and short sentences (High Burstiness)
    sample_text = """
    The quick brown fox jumps over the lazy dog. It was a beautifully sunny day! 
    Suddenly, out of absolutely nowhere, a massive, dark thunderstorm rolled over the hills and drenched everything in sight without a moment's warning. 
    We ran. The fox ran.
    """
    
    print("Loading model and analyzing text...\n")
    
    burstiness = calculate_burstiness(sample_text)
    perplexity = calculate_perplexity(sample_text)
    
    print(f"--- Results ---")
    print(f"Burstiness Score: {burstiness:.2f} (Higher means more human-like variance)")
    print(f"Perplexity Score: {perplexity:.2f} (Higher means more human-like unpredictability)")
    
    # A very basic rule-of-thumb classifier
    print("\n--- Verdict ---")
    if perplexity < 40 and burstiness < 5:
        print("Conclusion: Highly likely to be AI-generated.")
    elif perplexity > 60 and burstiness > 8:
        print("Conclusion: Highly likely to be Human-written.")
    else:
        print("Conclusion: Mixed signals. Could be human-edited AI or a highly structured human writer.")