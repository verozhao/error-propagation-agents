# Human Annotation Protocol

## Overview

Annotators evaluate LLM pipeline outputs to validate automated metrics
(persistence, severity, quality scores) against human judgment.

Platform: Prolific
Target: 3 annotators per item, majority vote
Estimated time per HIT: 2-3 minutes
Payment: $15/hr effective rate

## 5-Point Quality Rubric

| Score | Label       | Definition                                                        |
|-------|-------------|-------------------------------------------------------------------|
| 5     | Excellent   | Fully correct, complete, well-structured answer                   |
| 4     | Good        | Mostly correct with minor omissions, no factual errors            |
| 3     | Acceptable  | Contains some correct information but also noticeable gaps        |
| 2     | Poor        | Contains factual errors or is substantially incomplete            |
| 1     | Unacceptable| Mostly wrong, incoherent, or completely off-topic                 |

## Error Persistence Column

For items where an error was injected, annotators also answer:

> "Is the following specific error still present in this output?"

The injected error description is shown (e.g., "The entity 'Paris' was
replaced with 'Berlin'"). Annotators respond:

- **Yes, clearly present**: The error appears verbatim or nearly so
- **Yes, partially present**: The error influenced the output but is
  restated differently
- **No, corrected**: The output does not contain the error
- **Cannot determine**: Insufficient information to judge

## Qualification Check

Before starting, annotators complete 5 calibration items with known
ground truth. Requirement: >= 4/5 correct on quality rubric (within
1 point of gold label) AND >= 4/5 correct on persistence judgment.

Annotators who fail qualification are excluded.

## Calibration Sample

10 items from the pilot sweep with high inter-annotator agreement
from the automated metrics serve as ongoing calibration checks
(inserted at ~10% rate). Annotators deviating by > 2 points on
quality or disagreeing on persistence on > 50% of calibration items
are flagged for review.

## Blinding

- Annotators do NOT know which model produced the output
- Annotators do NOT know whether the item had an error injected
- Annotators do NOT know which pipeline configuration was used
- Item order is randomized per annotator
- The persistence question is shown only for injected items (but
  annotators are told "some items may or may not have had an error
  introduced" to avoid tipping off the injection status)

## HIT Structure

Each HIT contains:

1. **Query**: The original question
2. **Pipeline output**: The compose-step output (not the verify meta-comment)
3. **Quality rating**: 5-point rubric (always shown)
4. **Error persistence**: Shown for injected items only, with error description
5. **Free-text comment**: Optional, for edge cases

## Payment Tracking

| Phase          | n_items | n_annotators | HITs_total | est_cost |
|----------------|---------|--------------|------------|----------|
| Qualification  | 5       | pool         | --         | $0       |
| Pilot (Day 5)  | 50      | 3            | 150        | ~$15     |
| Main (Day 6+)  | 200     | 3            | 600        | ~$60     |

## Inter-Rater Agreement

Report Krippendorff's alpha for:
- Quality rubric (ordinal)
- Error persistence (nominal, 4 categories)

Target: alpha >= 0.6 (moderate agreement) for quality,
alpha >= 0.7 (substantial) for persistence.

## Data Format

Annotations are stored as JSONL with fields:
```json
{
  "item_id": "trial_xxxx",
  "annotator_id": "anon_hash",
  "quality_score": 4,
  "persistence_judgment": "yes_partial",
  "comment": "",
  "timestamp": "2026-04-28T12:00:00Z",
  "is_calibration": false
}
```
