# Fashion Recommendation Engine

Personalised top-10 recommendations for a (fictional) Noida ethnic-wear chain: 35 stores + online, ~1.15M raw event rows. Replaces the one-size-fits-all "popular items" shelf.

## Result (held-out last 60 days, 19,486 shoppers)

| Success criterion | Target | Achieved |
|---|---|---|
| NDCG@10 vs popularity | >= +25% | **+193%** (0.0302 -> 0.0885) |
| Catalogue coverage | >= 40% | **43.5%** |

|                                   |   precision@10 |   recall@10 |   ndcg@10 |   coverage |   diversity |   new_item_recall |
|:----------------------------------|---------------:|------------:|----------:|-----------:|------------:|------------------:|
| popularity                        |         0.0124 |      0.0454 |    0.0302 |     0.0227 |      0.7209 |            0      |
| content_based                     |         0.0115 |      0.0349 |    0.024  |     1      |      0.4215 |            0.0418 |
| item_item_cf                      |         0.033  |      0.1102 |    0.0782 |     0.5427 |      0.7186 |            0.0001 |
| als                               |         0.0328 |      0.1069 |    0.0763 |     0.4153 |      0.7342 |            0.0001 |
| hybrid (no re-rank)               |         0.0386 |      0.125  |    0.0882 |     0.4467 |      0.6876 |            0.0657 |
| hybrid (tuned: pop-penalty + MMR) |         0.0387 |      0.1256 |    0.0885 |     0.4353 |      0.7062 |            0.0634 |

By shopper type (heavy = top quartile of history, light = bottom quartile, cold = no history before the cutoff):

| model                             | group   |   users |   ndcg@10 |   recall@10 |   coverage |
|:----------------------------------|:--------|--------:|----------:|------------:|-----------:|
| popularity                        | heavy   |    4646 |    0.0219 |      0.0282 |     0.0227 |
| popularity                        | mid     |    8777 |    0.0281 |      0.0432 |     0.0133 |
| popularity                        | light   |    4504 |    0.0343 |      0.0528 |     0.0093 |
| popularity                        | cold    |    1559 |    0.0558 |      0.0871 |     0.0067 |
| als                               | heavy   |    4646 |    0.0957 |      0.1101 |     0.3813 |
| als                               | mid     |    8777 |    0.0816 |      0.1205 |     0.3727 |
| als                               | light   |    4504 |    0.0717 |      0.1122 |     0.3753 |
| als                               | cold    |    1559 |    0.0024 |      0.006  |     0.0133 |
| hybrid (tuned: pop-penalty + MMR) | heavy   |    4646 |    0.1073 |      0.125  |     0.332  |
| hybrid (tuned: pop-penalty + MMR) | mid     |    8777 |    0.0903 |      0.1327 |     0.3587 |
| hybrid (tuned: pop-penalty + MMR) | light   |    4504 |    0.0763 |      0.1239 |     0.4187 |
| hybrid (tuned: pop-penalty + MMR) | cold    |    1559 |    0.0576 |      0.0924 |     0.0067 |

Takeaways: collaborative models shine for heavy users; light users lean on content; cold-start users fall back to popularity (content/CF have nothing to work with), and new items are only reachable through the content path (`new_item_recall`).

## Accuracy vs diversity trade-off (what I chose and why)
Two knobs sit on top of the blended score: a **popularity penalty** (promote long-tail items) and **MMR re-ranking** (penalise a pick that is similar to earlier picks in the list). Test-window sweep:

| pop penalty | MMR lambda | NDCG@10 | coverage | diversity |
|---|---|---|---|---|
| 0 | 0 | 0.0882 | 44.7% | 0.688 |
| **0 (chosen)** | **0.15** | **0.0885** | **43.5%** | **0.706** |
| 0 | 0.5 | 0.0849 | 43.1% | 0.772 |
| 0.2 | 0.15 | 0.0930 | 38.7% | 0.688 |
| 0.3 | 0 | 0.0917 | 34.9% | 0.667 |

Rule, applied on a **validation** window (not the test set): maximise NDCG subject to coverage >= 42% (target 40% + safety margin) and intra-list diversity >= 0.70. Light MMR (0.15) lifts diversity from 0.69 to 0.71 for essentially no NDCG cost, whereas heavy diversification (lambda 0.5) costs ~4% NDCG for variety shoppers may not value. A bigger popularity penalty raises new-item discovery but drops coverage here because the hybrid already boosts thin-history items, so more penalty just swaps in noise. Both knobs are live sliders in the demo so merchandisers can move along the curve.

## Pipeline
1. **Data** (`fashionrec/data_gen.py`): synthetic, with the quirks from the brief: duplicates, null users, mixed event casing, `S/Small/36` sizes, missing colour/fabric, negative stock. Taste is latent (segments + a hidden item factor attributes can't see), so content and CF each capture part of the signal.
2. **Cleaning** (`prep.py`, report in `results/cleaning_report.json`): dedupe, normalise events and sizes (16 -> 5 labels), recover colour/fabric from product titles where possible else `Unknown`, flag + clip negative stock (item stock gates what is recommended at serve time).
3. **Interactions**: user x item matrix, weights view 1 / cart 3 / purchase 5 / return -6 (capped counts, clipped at 0).
4. **Split by time**: train < day 245, validation 245-305 (tuning), test >= 305 (final model refit on < 305). Only items *new to the user* count as relevant, so there is no leakage and no "recommend what they already bought".
5. **Models** (`models.py`): popularity, content-based (attribute profile cosine), item-item CF (log-weighted cosine, shrinkage, top-80 neighbours), implicit ALS (own NumPy implementation, 16 factors), **hybrid** (z-scored blend; weights depend on user history; items with <5 interactions swap their CF signal for content + a new-arrival boost; cold users get popularity).
6. **Metrics** (`metrics.py`): precision@10, recall@10, NDCG@10, catalogue coverage, intra-list diversity (1 - mean pairwise content similarity), new-item recall.
7. **Serve**: `fashionrec/serve.py` (top-10 + a reason per item: "Because you bought X", "Matches your taste for Silk & Festive", "New arrival...", "Trending..."), `api.py` (FastAPI), `app.py` (Streamlit).
8. **A/B plan**: `results/ab_test_plan.md` (stretch).

## Run it
```bash
pip install -r requirements.txt
python -m fashionrec.train          # generate data, clean, tune, evaluate, write results/ + artifacts/bundle.pkl (~2 min)
uvicorn api:app --reload            # GET /recommend/{user_id}?k=10&model=hybrid   (docs at /docs)
streamlit run app.py
```
`notebook.ipynb` walks through the same steps.

## Honest caveats
- **Synthetic data.** Lifts here are inflated vs. real life (taste signal is strong by construction). Treat the numbers as proof the pipeline works, not as a forecast; re-run on real logs or a public set such as H&M.
- **LightFM / `implicit` not used**: no internet in my build environment, so ALS is hand-written (same objective as `implicit`; swap it in for speed on real data).
- `api.py` and `app.py` could not be executed where I built this (FastAPI/Streamlit not installable offline); the `Recommender` class behind both was tested directly. Expect to fix at most small import/typing issues.
- Items launched after the training cutoff are scored at eval time using only their attributes (a simplification of a real launch-day flow).
