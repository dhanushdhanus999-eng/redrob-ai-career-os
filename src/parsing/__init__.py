"""Structured parsers used by the Phase 2 baseline pipeline."""

from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser

__all__ = ["CandidateProfileParser", "JobDescriptionParser"]
