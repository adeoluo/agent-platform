"""A small fixed corpus and question set used by the demo, the eval suite, and tests.

The content describes a fictional company and product. It is deliberately small so results
are easy to audit by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

ACME_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "refund-policy",
        "text": "Customers may request a full refund within 30 days of purchase. Refunds are "
        "issued to the original payment method within 5 business days. Opened hardware is "
        "subject to a 15 percent restocking fee.",
    },
    {
        "id": "shipping",
        "text": "Standard shipping takes 4 to 6 business days within the continental United "
        "States. Express shipping arrives in 2 business days and costs 18 dollars. Orders over "
        "75 dollars qualify for free shipping.",
    },
    {
        "id": "warranty",
        "text": "Every Lumen sensor includes a two year limited warranty covering manufacturing "
        "defects. The warranty does not cover water damage or physical abuse. Warranty claims "
        "require the device serial number.",
    },
    {
        "id": "support-hours",
        "text": "The support team is available Monday through Friday from 8 am to 6 pm Central "
        "Time. Priority support customers can reach an on-call engineer 24 hours a day. The "
        "average first response time is under 2 hours.",
    },
    {
        "id": "security",
        "text": "All customer data is encrypted at rest with AES-256. Data in transit is "
        "protected with TLS 1.3. Access to production systems requires hardware security keys.",
    },
    {
        "id": "product-lumen",
        "text": "The Lumen sensor measures temperature, humidity, and air quality every 60 "
        "seconds. The Lumen battery lasts up to 18 months on a single charge. The sensor "
        "connects over Wi-Fi or Thread.",
    },
    {
        "id": "pricing",
        "text": "The Starter plan costs 9 dollars per month and supports up to 5 sensors. The "
        "Team plan costs 49 dollars per month and supports up to 50 sensors. Enterprise "
        "pricing is negotiated annually.",
    },
    {
        "id": "data-retention",
        "text": "Raw sensor readings are retained for 13 months. Aggregated hourly summaries "
        "are retained indefinitely. Customers can export their data as CSV at any time.",
    },
]

GLOBEX_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "globex-handbook",
        "text": "Globex employees receive 25 days of paid vacation per year. Expense reports "
        "are due by the fifth day of each month. The Globex office is closed on public "
        "holidays.",
    },
]


@dataclass(frozen=True)
class EvalCase:
    """One evaluation question with its expected answer substring and source."""

    question: str
    target: str
    kind: str  # "retrieval", "arithmetic", or "unanswerable"
    doc_id: str | None = None


NOT_FOUND_TARGET = "could not find"

EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "How many days do customers have to request a refund?",
        "30 days",
        "retrieval",
        "refund-policy",
    ),
    EvalCase(
        "How long does standard shipping take?", "4 to 6 business days", "retrieval", "shipping"
    ),
    EvalCase("How much does express shipping cost?", "18 dollars", "retrieval", "shipping"),
    EvalCase(
        "What is the restocking fee for opened hardware?",
        "15 percent",
        "retrieval",
        "refund-policy",
    ),
    EvalCase("How long is the Lumen warranty?", "two year", "retrieval", "warranty"),
    EvalCase(
        "Does the warranty cover water damage?",
        "does not cover water damage",
        "retrieval",
        "warranty",
    ),
    EvalCase(
        "When is the support team available?", "Monday through Friday", "retrieval", "support-hours"
    ),
    EvalCase("What encryption protects customer data at rest?", "AES-256", "retrieval", "security"),
    EvalCase("Which protocol protects data in transit?", "TLS 1.3", "retrieval", "security"),
    EvalCase("How long does the Lumen battery last?", "18 months", "retrieval", "product-lumen"),
    EvalCase(
        "How often does the sensor measure air quality?",
        "every 60 seconds",
        "retrieval",
        "product-lumen",
    ),
    EvalCase("How much does the Team plan cost per month?", "49 dollars", "retrieval", "pricing"),
    EvalCase(
        "How long are raw sensor readings retained?", "13 months", "retrieval", "data-retention"
    ),
    EvalCase("Which orders qualify for free shipping?", "75 dollars", "retrieval", "shipping"),
    EvalCase("How quickly are refunds issued?", "5 business days", "retrieval", "refund-policy"),
    EvalCase("What is 17 * 23?", "391", "arithmetic"),
    EvalCase("Calculate (120 - 45) / 3", "25", "arithmetic"),
    EvalCase("What is 2 ** 10?", "1024", "arithmetic"),
    EvalCase("Who founded the company?", NOT_FOUND_TARGET, "unanswerable"),
    EvalCase("What is the capital of Peru?", NOT_FOUND_TARGET, "unanswerable"),
]
