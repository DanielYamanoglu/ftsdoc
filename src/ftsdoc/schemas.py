"""Typed output schemas, one per document type."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    """An invoice."""
    invoice_number: Optional[str] = Field(None, description="Invoice identifier, e.g. INV-2024-00123")
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor: Optional[str] = Field(None, description="Company issuing the invoice")
    customer: Optional[str] = Field(None, description="Company being billed")
    contact_email: Optional[str] = None
    total: Optional[str] = Field(None, description="Total amount due, with currency")


class CV(BaseModel):
    """A curriculum vitae / resume."""
    name: Optional[str] = None
    email: Optional[str] = None
    organizations: list[str] = Field(default_factory=list, description="Employers and educational institutions")
    skills: list[str] = Field(default_factory=list)


class Contract(BaseModel):
    """A commercial contract."""
    effective_date: Optional[str] = None
    parties: list[str] = Field(default_factory=list, description="Contracting companies")
    fee: Optional[str] = Field(None, description="Fee payable, with currency")
    notice_email: Optional[str] = None
    signatories: list[str] = Field(default_factory=list)


class TechSpec(BaseModel):
    """A technical specification sheet."""
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    revision_date: Optional[str] = None
    list_price: Optional[str] = None
    support_email: Optional[str] = None
    specs: dict[str, str] = Field(default_factory=dict, description="Specification key/value pairs")


SCHEMAS: dict[str, type[BaseModel]] = {"invoice": Invoice, "cv": CV, "contract": Contract, "tech_spec": TechSpec}
DocType = Literal["invoice", "cv", "contract", "tech_spec"]


class ExtractionResult(BaseModel):
    source: str
    doc_type: str
    classifier_confidence: Optional[float] = None
    backend: str
    data: Union[Invoice, CV, Contract, TechSpec]
