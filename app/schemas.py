"""
Pydantic schemas for the document extraction pipeline.
"""
from pydantic import BaseModel
from typing import Optional


class MetadataExtraction(BaseModel):
    agreement_value: Optional[str] = None
    agreement_start_date: Optional[str] = None
    agreement_end_date: Optional[str] = None
    renewal_notice_days: Optional[str] = None
    party_one: Optional[str] = None
    party_two: Optional[str] = None
