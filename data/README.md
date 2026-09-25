# Data provenance and licensing

Nothing in `data/raw/` is committed. Run `python scripts/download_data.py` to
reproduce it; `data/processed/splits/` is committed so results are reproducible
without redistributing any data.

## Sources

| What | Source | Licence |
| --- | --- | --- |
| Eye-tracking (AiR-D) | [szzexpoi/AiR](https://github.com/szzexpoi/AiR), Chen, Jiang, Yang & Zhao, ECCV 2020 | see repository |
| Questions and answers (GQA) | [GQA](https://cs.stanford.edu/people/dorarad/gqa/), Hudson & Manning, CVPR 2019 | CC BY 4.0 |
| Images (Visual Genome) | Krishna et al., IJCV 2017 | CC BY 4.0 |

Visual Genome is CC BY 4.0, so SHAP overlays drawn on these images may be
published as long as attribution is given. Any publication or dashboard built
from this repository must credit Visual Genome, GQA and AiR-D.

## What the download actually fetches

Two bulk archives are deliberately **not** downloaded in full:

- `questions1.2.zip` (1.5 GB) contains every GQA split. The server supports byte
  ranges, so the zip's central directory is read over HTTP and only
  `val_balanced_questions.json` (114 MB) is extracted.
- `images.zip` (21.8 GB) contains all GQA images. AiR-D needs 987 of them, and
  GQA image IDs are Visual Genome image IDs, so the images are fetched
  individually from `VG_100K` with a `VG_100K_2` fallback (~100 MB).

## Layout after `download_data.py`

```
data/raw/
├── images/                         987 JPGs named {image_id}.jpg
├── gqa/val_balanced_questions.json 132,062 questions (AiR-D uses 1,422)
├── air_d/
│   ├── aggregated_maps/
│   │   ├── fixmaps/                1,454 PNGs, 256x256, grayscale
│   │   ├── fixmaps_corr/           934
│   │   └── fixmaps_incorr/         934
│   ├── temporal_maps/
│   │   └── fixmaps[_corr|_incorr]_{0s,1s,2s}/   1,422 (934 for corr/incorr)
│   └── scanpaths/
│       ├── fix/{qid}/{subject}.mat 1,422 questions x 10 subjects
│       └── consolidated_answers.json
└── manifest.json
```

## Things the upstream README does not tell you

These were established by inspecting the actual archives; the code depends on
them, so they are recorded here.

- **`fixmaps` holds 1,454 maps but only 1,422 questions are usable.** The other
  32 have no temporal bins, no scanpaths and no recorded answers.
  `load_air_d_qids()` returns the intersection.
- **Scanpaths are `fix/{qid}/{subject}.mat`**, one directory per question, not
  the flat `fix/{qid}_{subject}.mat` naming the paper's README suggests.
- **Ten subjects per question, twenty in total.** `consolidated_answers.json`
  has one key per participant (`AL`, `AR`, ..., `ZL`), and each question was
  seen by ten of them, which is why `accuracy` is always a multiple of 0.1.
- **`consolidated_answers.json` is keyed by field, not by question:** top-level
  keys are `accuracy`, `question`, `answer` and the twenty participant initials,
  each mapping question ID to a value.
- **479 questions were answered correctly by everyone**, 934 are mixed and 9 were
  missed by everyone. `fixmaps_corr` and `fixmaps_incorr` exist for the 934
  mixed ones only.
- **`.mat` fields:** `xy` is `(n_fixations, 2)` float64 in *original image pixel
  coordinates*, `t` is `(n_fixations, 2)` uint16 holding fixation start and end
  in milliseconds within the 3,000 ms trial.

## Hold-out set

AiR-D defines a hold-out split (319 images / 406 questions), but its
eye-tracking annotations are private and the two Drive archives contain only GQA
IDs and question-answer pairs. Issue #5 in the upstream repository promises a
release; as of this writing it has not happened. The download treats it as
optional and the default splits do not use it.
