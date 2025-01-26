import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_answers(raw_answer, expected_order):
    answer_dict = {}
    for category in expected_order:
        patterns = [
            rf"{category}[\s:]*\[?(YES|NO|MAYBE|Y|N|M)\]?",
            rf"{category.replace('_', ' ')}[\s:]*\[?(YES|NO|MAYBE|Y|N|M)\]?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_answer, re.IGNORECASE)
            if match:
                raw_result = match.group(1).upper()
                answer = "YES" if raw_result in ("Y", "YES") else "NO" if raw_result in ("N", "NO") else "MAYBE"
                answer_dict[category] = answer
                break
        else:
            answer_dict[category] = "NO"
    
    return [answer_dict[cat] for cat in expected_order]

def analyze_script(script):
    logger.info("Starting Analysis")
    start_time = time.time()
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        
        max_chunk_size = 2048  # Increased chunk size
        chunks = [script[i:i+max_chunk_size] for i in range(0, len(script), max_chunk_size)]
        
        expected_order = [
            "VIOLENCE", "DEATH", "SUBSTANCE_USE", "GORE", 
            "VOMIT", "SEXUAL_CONTENT", "SEXUAL_ABUSE", 
            "SELF_HARM", "GUN_USE", "ANIMAL_CRUELTY", "MENTAL_HEALTH"
        ]
        
        identified = {cat: 0 for cat in expected_order}
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {chunk_idx}/{len(chunks)}")
            
            prompt = f"""Comprehensive Sensitive Content Analysis Protocol

You are a highly specialized content analysis AI. Your task is to meticulously examine the provided text for potentially sensitive or harmful content. Perform an exhaustive, multi-layered analysis with extreme precision.

ANALYSIS DIRECTIVES:
1. Thoroughly scan the entire text
2. Provide detailed rationale for each classification
3. Flag ANY potential indicators of sensitive content
4. Be extremely conservative in your assessments

CONTENT CATEGORIES FOR DETAILED EXAMINATION:

VIOLENCE: 
- Comprehensive assessment of physical aggression
- Includes: fighting, torture, physical confrontations
- Evaluate intensity, graphic nature, psychological impact
- Detailed description of violent elements

DEATH: 
- Comprehensive analysis of fatal incidents
- Includes: killing descriptions, death circumstances
- Assess psychological and emotional context
- Evaluate graphic or traumatic elements

SUBSTANCE_USE:
- Detailed examination of drug/alcohol references
- Include: misuse, addiction patterns, context
- Assess severity and potential psychological implications

GORE:
- Thorough investigation of graphic bodily descriptions
- Includes: blood, injury, extreme physical trauma
- Evaluate graphical intensity and psychological impact

VOMIT:
- Detailed analysis of bodily fluid descriptions
- Assess context, intensity, psychological implications

SEXUAL_CONTENT:
- Comprehensive sexual activity assessment
- Evaluate explicit vs. implied content
- Analyze psychological and contextual elements

SEXUAL_ABUSE:
- Rigorous examination of non-consensual sexual acts
- Detailed assessment of power dynamics
- Psychological impact analysis

SELF_HARM:
- Thorough investigation of suicide/self-injury references
- Comprehensive psychological risk assessment
- Evaluate implicit and explicit indicators

GUN_USE:
- Detailed firearms and gun violence analysis
- Assess context, threat level, psychological impact

ANIMAL_CRUELTY:
- Comprehensive animal harm investigation
- Evaluate physical and psychological dimensions

MENTAL_HEALTH:
- Extensive psychological distress examination
- Detailed analysis of emotional and mental states
- Assess severity and potential risk indicators

ANALYSIS TEXT:
{chunk}

RESPONSE FORMAT:
For each category, provide:
1. YES/NO classification
2. Brief substantive explanation
3. Confidence level (low/medium/high)"""

            inputs = tokenizer(prompt, return_tensors="pt", 
                               truncation=True, 
                               max_length=4096).to(model.device)
            
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                max_new_tokens=1024,  # Increased tokens
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            input_length = inputs.input_ids.size(1)
            generated_tokens = outputs[0][input_length:]
            raw_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            logger.info("[Model Raw Response]")
            logger.info(raw_answer)
            
            answers = extract_answers(raw_answer, expected_order)
            
            logger.info("[Analysis Results]")
            for cat, ans in zip(expected_order, answers):
                logger.info(f"{cat}: {ans}")
                if ans == "YES":
                    identified[cat] += 1
        
        logger.info("\n=== Final Results ===")
        for cat in expected_order:
            score = identified[cat]
            status = "CONFIRMED" if score > 0 else "NOT FOUND"
            logger.info(f"{cat}: {status} ({score}/{len(chunks)} chunks)")
        
        total_time = time.time() - start_time
        logger.info(f"Total analysis time: {total_time:.1f}s")
        return identified
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": str(e)}

def get_detailed_analysis(script):
    return analyze_script(script)