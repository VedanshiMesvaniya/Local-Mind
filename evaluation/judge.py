from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from config.settings import UTILITY_MODEL, OLLAMA_HOST

JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are a strict grading AI. Compare the ACTUAL ANSWER to the EXPECTED ANSWER.
Ignore differences in formatting or wording. Focus purely on factual accuracy and completeness.

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}

Reply with ONLY a number from 0 to 10 representing the score. 
10 = Perfect match. 0 = Completely wrong or missing key facts.
""")

def grade_with_llm(expected: str, actual: str) -> float:
    # Truncate actual answer to avoid context limits
    actual_truncated = actual[:2000] 
    
    llm = ChatOllama(model=UTILITY_MODEL, temperature=0, base_url=OLLAMA_HOST, options={"think": False})
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    
    try:
        score_text = chain.invoke({"expected": expected, "actual": actual_truncated})
        # Extract the number from the response
        match = re.search(r'\d+(\.\d)?', score_text)
        return float(match.group()) if match else 0.0
    except Exception as e:
        print(f"Judge error: {e}")
        return 0.0