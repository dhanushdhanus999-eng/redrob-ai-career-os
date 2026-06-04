# Demo Video Script

Target length: 5 to 7 minutes.

## 0:00-0:45 - Hook

India Runs Track 1 asks us to rank the best candidates from a 100,000-profile
pool for a very specific Senior AI Engineer role at Redrob AI. Keyword filters
miss the real signal here: production retrieval experience, ranking evaluation
judgment, activity, availability, and evidence that a candidate can build
systems rather than demos.

This project builds an AI recruiting pipeline that turns a long nuanced JD into
a ranked shortlist with score breakdowns and explanations.

## 0:45-2:00 - Architecture Walkthrough

Show `docs/architecture_diagram.png`.

The pipeline starts by parsing the job description into structure: title,
seniority, skills, experience, location, and responsibilities. Retrieval then
uses lexical and dense search so exact terms like Qdrant and NDCG are not lost,
while semantic matches can still surface strong adjacent candidates.

The feature layer separates semantic, skill, experience, and behavioral signals.
That matters because Redrob's brief explicitly calls out subtle behavioral
signals, so the ranker treats recency, response rate, open-to-work state,
profile completeness, and recruiter engagement as first-class evidence.

## 2:00-4:00 - Live Demo

Open the Gradio app.

Run the default Senior AI Engineer JD first. Point out the score breakdown chart
and the ranked table. For one top candidate, read the rationale and call out the
matched skills, missing skills, experience fit, and behavioral confidence.

Then paste a different JD, such as a Marketing Manager or Backend Platform
Engineer role. Show that the ranked candidates and score components shift,
which demonstrates that the demo is responding to the JD rather than displaying
a fixed shortlist.

## 4:00-5:30 - Results and Evidence

Open the README results section. Explain that the released public bundle has no
ground-truth labels, so NDCG, MAP, and precision are hidden-eval metrics rather
than local numbers.

Show the submission contract: exactly 100 rows, required columns in order, and
non-increasing scores. Then show the performance profile and the feature signal
chart to demonstrate that the pipeline is implemented and label-aware rather
than metric-fabricated.

## 5:30-6:30 - Close

Close by emphasizing the three strongest points:

1. The system uses the real released 100,000-candidate pool.
2. Behavioral signals are modeled beside text relevance.
3. The code is ready for supervised LTR and reranking once labels or official
   feedback are available.

Suggested closing line:

This is not just candidate search; it is a transparent ranking system built for
how modern recruiters actually decide fit.
