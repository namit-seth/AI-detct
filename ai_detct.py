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
    Getting a neural network to run on a microcontroller is harder than it sounds.
You are not just squeezing a model into less memory — you are fighting the compiler,
the scheduler, the flash budget, and sometimes the hardware itself, all at once. This
paper looks at three studies published in 2025 that each attack a different piece
of that problem. The first, Ariel-ML [1], builds a Rust-based platform that splits
inference work across multiple CPU cores on low-power MCUs. The second [2] puts
a voice-command classifier on an Arduino Nano with 256 KB of RAM — it handles
23 keywords. The third [3] runs compressed MobileNetV2 variants on a Cortex-M4
for person detection, then measures Wi-Fi transmission energy on top of that. None
of these papers talks to the other two. Reading them together, though, a clear
picture emerges: 8-bit quantization is the one technique that always works, multi-
core scheduling is being completely ignored by most frameworks, and the field still
has no agreed way to measure and report energy consumption. This paper maps
those findings, identifies the gaps, and proposes a practical six-step workflow for
actually getting TinyML models deployed on real embedded hardware.
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