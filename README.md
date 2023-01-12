---
annotations_creators:
- crowdsourced

language:
- ja

language_creators:
- crowdsourced

license:
- unknown

multilinguality:
- monolingual

pretty_name: wrime

tags:
- sentiment-analysis
- wrime

task_categories:
- text-classification
task_ids:
- sentiment-classification

datasets:
- ver1
- ver2

metrics:
- accuracy
---

# Dataset Card for WRIME

[![CI](https://github.com/shunk031/huggingface-datasets_wrime/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_wrime/actions/workflows/ci.yaml)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- Homepage: https://github.com/ids-cv/wrime
- Repository: https://github.com/shunk031/huggingface-datasets_wrime
- Paper: https://aclanthology.org/2021.naacl-main.169/

### Dataset Summary

[More Information Needed]

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

[More Information Needed]

## Dataset Structure

### Data Instances

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/wrime", name="ver1")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence', 'user_id', 'datetime', 'writer', 'reader1', 'reader2', 'reader3', 'avg_readers'],
#         num_rows: 40000
#     })
#     validation: Dataset({
#         features: ['sentence', 'user_id', 'datetime', 'writer', 'reader1', 'reader2', 'reader3', 'avg_readers'],
#         num_rows: 1200
#     })
#     test: Dataset({
#         features: ['sentence', 'user_id', 'datetime', 'writer', 'reader1', 'reader2', 'reader3', 'avg_readers'],
#         num_rows: 2000
#     })
# })
```

An example of looks as follows:

```json
{
    "sentence": "ぼけっとしてたらこんな時間｡チャリあるから食べにでたいのに…",
    "user_id": "1",
    "datetime": "2012/07/31 23:48",
    "writer": {
        "joy": 0,
        "sadness": 1,
        "anticipation": 2,
        "surprise": 1,
        "anger": 1,
        "fear": 0,
        "disgust": 0,
        "trust": 1
    },
    "reader1": {
        "joy": 0,
        "sadness": 2,
        "anticipation": 0,
        "surprise": 0,
        "anger": 0,
        "fear": 0,
        "disgust": 0,
        "trust": 0
    },
    "reader2": {
        "joy": 0,
        "sadness": 2,
        "anticipation": 0,
        "surprise": 1,
        "anger": 0,
        "fear": 0,
        "disgust": 0,
        "trust": 0
    },
    "reader3": {
        "joy": 0,
        "sadness": 2,
        "anticipation": 0,
        "surprise": 0,
        "anger": 0,
        "fear": 1,
        "disgust": 1,
        "trust": 0
    },
    "avg_readers": {
        "joy": 0,
        "sadness": 2,
        "anticipation": 0,
        "surprise": 0,
        "anger": 0,
        "fear": 0,
        "disgust": 0,
        "trust": 0
    }
}
```

### Data Fields

- `sentence`
- `user_id`
- `datetime`
- `writer`
    - `joy`
    - `sadness`
    - `anticipation`
    - `surprise`
    - `anger`
    - `fear`
    - `disgust`
    - `trust`
- `reader1`
    - `joy`
    - `sadness`
    - `anticipation`
    - `surprise`
    - `anger`
    - `fear`
    - `disgust`
    - `trust`
- `reader2`
    - `joy`
    - `sadness`
    - `anticipation`
    - `surprise`
    - `anger`
    - `fear`
    - `disgust`
    - `trust`
- `reader3`
    - `joy`
    - `sadness`
    - `anticipation`
    - `surprise`
    - `anger`
    - `fear`
    - `disgust`
    - `trust`
- `avg_readers`
    - `joy`
    - `sadness`
    - `anticipation`
    - `surprise`
    - `anger`
    - `fear`
    - `disgust`
    - `trust`

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

[More Information Needed]

### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.
