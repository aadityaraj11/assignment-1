"""
app/llm/extractor.py
====================
Constructs semantic few-shot prompts mapping generic instructions.
"""

import json
from pydantic import BaseModel

from app.retrieval.training_data import TrainingExample
from app.schemas import MetadataExtraction

# The base instructions providing definitions for LLM semantic capabilities without regexes
SYSTEM_PROMPT = """You extract structured metadata from legal rental documents.

FIELD DEFINITIONS:
- agreement_value: Identify the monetary amount represented as the primary rental/agreement value. Do NOT confuse it with security deposit, advance deposit, maintenance, utilities, penalties, insurance, or other charges.
- agreement_start_date: Identify the actual commencement/effective date of the agreement. Do NOT automatically return the document execution/signing date.
- agreement_end_date: Identify the actual expiry/end date of the agreement.
- renewal_notice_days: Identify the notice period relevant to renewal/termination. Use the full document context.
- party_one: Identify the first contractual party (usually the owner/lessor).
- party_two: Identify the second contractual party (usually the tenant/lessee).

Return JSON ONLY. Do not include chain-of-thought. Do not ask to explain reasoning. Do not output prose.
"""

def _format_example(ex: TrainingExample, index: int) -> str:
    # Build target JSON using schema to ensure key matching
    target_obj = MetadataExtraction(**ex.metadata).model_dump()
    target_json = json.dumps(target_obj, indent=2)
    
    return f"""EXAMPLE {index}:
DOCUMENT:
{ex.document_text}

TARGET:
{target_json}
"""

def build_extraction_prompt(
    target_document_text: str,
    retrieved_examples: list[TrainingExample]
) -> str:
    """
    Constructs the full retrieval-augmented prompt.
    """
    blocks = [SYSTEM_PROMPT]
    
    for i, ex in enumerate(retrieved_examples, start=1):
        blocks.append(_format_example(ex, i))
        
    blocks.append(f"""NEW DOCUMENT:
{target_document_text}

Remember: RETURN JSON ONLY matching the metadata schema exactly.""")

    return "\n\n".join(blocks)
