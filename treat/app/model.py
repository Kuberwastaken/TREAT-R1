from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from datetime import datetime
import time

def extract_answers(raw_answer, expected_order):
    """Robust parser with duplicate handling"""
    # Remove duplicate lines and normalize
    seen = set()
    clean_lines = []
    for line in raw_answer.split("\n"):
        normalized = line.upper().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            clean_lines.append(normalized)
    
    clean_text = " ".join(clean_lines)
    
    category_map = {cat.upper().replace("_", " "): cat for cat in expected_order}
    
    # Strict pattern matching
    pattern = r"\b({})\b\s*[:=]\s*\[?(YES|NO|MAYBE|Y|N|M)\]?".format("|".join(category_map.keys()))
    matches = re.findall(pattern, clean_text, re.IGNORECASE)
    
    answer_dict = {}
    for match in matches:
        category = category_map[match[0].upper()]
        answer = "YES" if match[1].upper() in ("Y", "YES") else "NO" if match[1].upper() in ("N", "NO") else "MAYBE"
        answer_dict[category] = answer
    
    return [answer_dict.get(cat, "NO") for cat in expected_order]

def analyze_script(script):
    print("\n=== Starting Analysis ===")
    start_time = time.time()
    
    try:
        # Model configuration
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        
        # Optimized chunking for 2k context window
        max_chunk_size = 1024  # Leaves 1000 tokens for response
        overlap = 128
        chunks = [script[i:i+max_chunk_size] for i in range(0, len(script), max_chunk_size - overlap)]
        
        expected_order = [
            "VIOLENCE", "DEATH", "SUBSTANCE_USE", "GORE", "VOMIT",
            "SEXUAL_CONTENT", "SEXUAL_ABUSE", "SELF_HARM",
            "GUN_USE", "ANIMAL_CRUELTY", "MENTAL_HEALTH"
        ]
        
        identified = {cat: 0 for cat in expected_order}
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            print(f"\nProcessing chunk {chunk_idx}/{len(chunks)}")
            print("=" * 50)
            
            # Optimized prompt with response space
            prompt = f"""TEXT ANALYSIS:
Respond ONLY with this exact format:

VIOLENCE: [YES/NO]
DEATH: [YES/NO]
SUBSTANCE_USE: [YES/NO]
GORE: [YES/NO]
VOMIT: [YES/NO]
SEXUAL_CONTENT: [YES/NO]
SEXUAL_ABUSE: [YES/NO]
SELF_HARM: [YES/NO]
GUN_USE: [YES/NO]
ANIMAL_CRUELTY: [YES/NO]
MENTAL_HEALTH: [YES/NO]

Text: {chunk[:768]}..."""  # Reduced text preview

            inputs = tokenizer(prompt, return_tensors="pt", 
                             truncation=True, 
                             max_length=1536,  # Leaves 512 tokens for response
                             padding_side="left").to(model.device)
            
            # Generation with controlled output
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Full response decoding
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean response extraction
            response_start = full_response.find("Text:") + len("Text:")
            raw_answer = full_response[response_start:].split("...")[-1].strip()
            
            print("\n[Model Raw Response]")
            print("-" * 50)
            print(raw_answer)
            print("-" * 50)
            
            # Parse and display
            answers = extract_answers(raw_answer, expected_order)
            
            print("\n[Analysis Results]")
            print("Category          | Status")
            print("------------------|-------")
            for cat, ans in zip(expected_order, answers):
                print(f"{cat:<17}| {ans}")
                if ans == "YES":
                    identified[cat] += 1
            
            print(f"\nChunk processed in {time.time()-chunk_start:.1f}s")
        
        # Final results
        print("\n\n=== Final Results ===")
        print("Category          | Confidence")
        print("------------------|-----------")
        for cat in expected_order:
            score = identified[cat]
            status = "CONFIRMED" if score > 0 else "NOT FOUND"
            print(f"{cat:<17}| {status} ({score}/{len(chunks)} chunks)")
        
        print(f"\nTotal analysis time: {time.time()-start_time:.1f}s")
        return identified
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        return {"error": str(e)}

def get_detailed_analysis(script):
    return analyze_script(script)