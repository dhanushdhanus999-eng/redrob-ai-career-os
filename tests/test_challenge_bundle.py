"""Unit tests for public challenge-bundle helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from src.data.challenge_bundle import (
    discover_challenge_bundle,
    flatten_candidate_record,
    parse_job_description,
    read_docx_text,
    stream_candidate_records,
)


def _make_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>""",
        )
        body = "".join(
            f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
        )
        archive.writestr(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}</w:body>
</w:document>""",
        )


class ChallengeBundleTests(unittest.TestCase):
    def test_parse_job_description(self) -> None:
        text = "\n".join(
            [
                "Job Description: Senior AI Engineer — Founding Team",
                "Company: Redrob AI",
                "Location: Pune/Noida, India",
                "Employment Type: Full-time",
                "Experience Required: 5-9 years",
                "Things you absolutely need",
                "Production retrieval experience",
                "Things we'd like you to have but won't reject you for",
                "Learning-to-rank background",
            ]
        )

        parsed = parse_job_description(text)
        self.assertEqual(parsed["company"], "Redrob AI")
        self.assertEqual(parsed["min_experience"], 5.0)
        self.assertIn("Production retrieval experience", str(parsed["required_skills"]))

    def test_flatten_candidate_record(self) -> None:
        record = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "headline": "AI Engineer",
                "summary": "Built ranking systems.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 6.5,
                "current_title": "Senior ML Engineer",
                "current_company": "Acme",
                "current_company_size": "201-500",
                "current_industry": "Software",
            },
            "career_history": [
                {
                    "company": "Acme",
                    "title": "Senior ML Engineer",
                    "description": "Built ranking systems.",
                }
            ],
            "education": [{"degree": "B.Tech", "field_of_study": "CSE", "institution": "ABC"}],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 48}
            ],
            "certifications": [{"name": "AWS", "issuer": "Amazon", "year": 2024}],
            "languages": [{"language": "English", "proficiency": "professional"}],
            "redrob_signals": {
                "profile_completeness_score": 90,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-01-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 12,
                "applications_submitted_30d": 3,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 12,
                "skill_assessment_scores": {"Python": 95},
                "connection_count": 500,
                "endorsements_received": 20,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 18, "max": 24},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 75,
                "search_appearance_30d": 40,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.5,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }

        flattened = flatten_candidate_record(record)
        self.assertEqual(flattened["candidate_id"], "CAND_0000001")
        self.assertEqual(flattened["skill_count"], 1)
        self.assertEqual(flattened["skill_assessment_count"], 1)
        self.assertIn("Python", flattened["skills"])

    def test_discover_bundle_and_stream_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "bundle"
            root.mkdir(parents=True)
            (root / "candidate_schema.json").write_text("{}", encoding="utf-8")
            (root / "sample_submission.csv").write_text(
                "candidate_id,rank,score,reasoning\nCAND_0000001,1,1.0,test\n",
                encoding="utf-8",
            )
            (root / "candidates.jsonl").write_text(
                json.dumps({"candidate_id": "CAND_0000001", "profile": {}, "career_history": [], "education": [], "skills": [], "redrob_signals": {}})
                + "\n",
                encoding="utf-8",
            )

            for name in (
                "job_description.docx",
                "README.docx",
                "redrob_signals_doc.docx",
                "submission_spec.docx",
            ):
                _make_minimal_docx(root / name, [name])

            bundle = discover_challenge_bundle(search_dirs=[root])
            self.assertEqual(bundle.root, root)
            self.assertEqual(read_docx_text(bundle.job_description), "job_description.docx")

            rows = list(stream_candidate_records(bundle.candidates))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_id"], "CAND_0000001")


if __name__ == "__main__":
    unittest.main()
