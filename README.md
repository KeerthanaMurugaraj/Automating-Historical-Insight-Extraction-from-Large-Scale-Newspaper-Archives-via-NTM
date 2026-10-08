# Automating Historical Insight Extraction from Large-Scale Newspaper Archives using BERTopic

Historical newspaper archives are hard to analyse at scale: topics shift over decades, OCR introduces noise, and the volume of text rules out manual reading. This project uses [BERTopic](https://github.com/MaartenGr/BERTopic), an embedding-based topic-modeling approach, to extract coherent and interpretable themes from such archives and to trace how they change over time.

The study covers a large collection of historical newspaper articles from the [impresso](https://impresso-project.ch/) project. The articles are translated to English, split into four time periods (1955–1970, 1971–1986, 1987–2002, 2003–2018) and embedded with long-context sentence-embedding models (GTE and Jina). On top of these embeddings we:

- build **static topic models** to obtain a global view of the themes in the corpus,
- run a **topics-over-time analysis** to see when themes emerge, peak and decline,
- **compare BERTopic with classical baselines** (LDA and NMF, including TF-IDF and named-entity variants) on topic coherence, topic diversity and runtime,
- study the effect of **hyperparameter tuning** and of the choice of embedding model.

## 📁 Repository Structure

```
.
├── README.md
├── LICENSE                          MIT License
├── requirements.txt                 Python dependencies
├── translate/                       Translation of the corpus to English (SLURM)
│   ├── translate_content.py
│   ├── translate_slurm.sh
│   └── translate.sh
└── notebooks/
    ├── Data_cleaning.ipynb          Cleaning before and after translation
    ├── Data_preparation_subsets/    Filtering, splitting into periods, preprocessing
    ├── Topic_modeling/              Static and dynamic topic modeling with BERTopic
    ├── evaluation/                  Code to train and score topic models
    └── Comparative_Study/
        ├── Hyperparameter_tuning/   BERTopic hyperparameter search
        └── model_comparison/        BERTopic vs. LDA, NMF and Top2Vec, per period
```

The steps are run in this order:

| Step | Where | Section |
|---|---|---|
| 1. Clean the raw export | `notebooks/Data_cleaning.ipynb` | Data Cleaning |
| 2. Translate to English | `translate/` | Translation |
| 3. Filter, split into periods and preprocess | `notebooks/Data_preparation_subsets/` | Data Preparation by Time Period |
| 4. Fit BERTopic and analyse topics over time | `notebooks/Topic_modeling/` | Topic Modeling with BERTopic |
| 5. Tune the BERTopic hyperparameters | `notebooks/Comparative_Study/Hyperparameter_tuning/` | BERTopic Hyperparameter Tuning |
| 6. Compare BERTopic with the baselines | `notebooks/Comparative_Study/model_comparison/` | Model Comparison |

Steps 4 to 6 all build on the preprocessed text of step 3. Steps 5 and 6 expect it as a dataset in OCTIS format, which is described in the BERTopic Hyperparameter Tuning section. The corpus itself is not included; see the Data Access section below.

## ⚙️ Installation

The experiments were run with Python 3.10. A GPU is strongly recommended for computing the document embeddings; the experiments in the paper used one NVIDIA A100 (40 GB).

```bash
git clone https://github.com/KeerthanaMurugaraj/Automating-Historical-Insight-Extraction-from-Large-Scale-Newspaper-Archives-via-NTM.git
cd Automating-Historical-Insight-Extraction-from-Large-Scale-Newspaper-Archives-via-NTM

conda create -n topic_modeling python=3.10
conda activate topic_modeling

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

[`requirements.txt`](requirements.txt) lists the packages the code imports. Versions are pinned where the version used in the experiments is known, among them `bertopic==0.16.3`, `octis==1.13.1` and `gensim==4.2.0`.

The notebooks expect the data under `datasets/` at the root of the repository and use relative paths. Adjust the paths if your data is stored elsewhere.

## 🧹 Data Cleaning

[`notebooks/Data_cleaning.ipynb`](notebooks/Data_cleaning.ipynb) prepares the raw Impresso export for topic modeling. It runs in two stages, with the translation step in between.

**Before translation.** The raw export is read as a `;`-delimited CSV with `title` and `content` columns. Each text is stripped of HTML markup (`<br>`, `<sup>`, other tags), unicode-normalised, and cleared of line breaks. The results are stored in two new columns, `title_clean` and `content_clean`, which are the input to the translation scripts.

**After translation.** The pickle files produced by the translation jobs are merged into a single DataFrame. The notebook then:

- detects the language of every translated document with `langdetect`, to find documents that were not translated to English,
- lowercases the text and removes quotes, backslashes, URLs and asterisks,
- collapses repeated letters (three or more) and repeated non-word characters, which are common OCR artifacts,
- removes punctuation other than full stops, commas, semicolons and question marks,
- normalises whitespace,
- saves the cleaned DataFrame as a pickle file.

The cleaned English text is stored in `content_translated_processed`.

Before running, set the three paths in the notebook: the raw CSV file, the directory with the translated pickle files, and the output pickle file.

## 🌍 Translation

The scripts in [`translate/`](translate/) translate the cleaned non-English texts to English with Google Translate. Because the corpus is large, the work is split into row ranges that run as parallel jobs on a SLURM cluster.

| File | Purpose |
|---|---|
| `translate_content.py` | Translates the selected columns for one row range of a pickled DataFrame |
| `translate_slurm.sh` | SLURM job that runs `translate_content.py` on one row range |
| `translate.sh` | Splits the corpus into row ranges and submits one SLURM job per range |

For each text, `translate_content.py` calls Google Translate and checks the result with `langdetect`. If the output is not detected as English, it retries up to five times and then logs the failure.

**Run on a SLURM cluster**

1. In `translate_slurm.sh`, set `DATA_DIR`, the input file name in `IP_FILE`, the columns to translate in `COLUMNS`, the path to this folder, and the name of your conda environment.
2. In `translate.sh`, set `NUM_ROWS` to the number of rows in your dataset and `NUM_MACHINES` to the number of parallel jobs.
3. Submit the jobs:

```bash
cd translate
bash translate.sh
```

**Run a single row range without SLURM**

```bash
python translate_content.py \
    -f 0 -l 1000 \
    -i /path/to/data.pkl \
    -o /path/to/output_dir \
    -t content_clean,title_clean
```

| Argument | Meaning |
|---|---|
| `-f`, `--first_row` | First row to translate |
| `-l`, `--last_row` | Last row to translate (exclusive) |
| `-i`, `--input_pkl_file` | Input pickle file containing the DataFrame |
| `-o`, `--output_directory` | Directory for the output file |
| `-t`, `--list_columns` | Comma-separated list of columns to translate |

Each run writes `<input_name>_<first_row>_<last_row>.pkl` to the output directory. The file contains the original rows plus one `<column>_translated` column per translated column, for example `content_clean_translated`. A log file `translate_<first_row>_<last_row>.out` is written for each SLURM job.

**Requirements:** `pandas`, `googletrans`, `langdetect`, `swifter`. The cleaning notebook additionally needs `beautifulsoup4`.

## 🗂️ Data Preparation by Time Period

The notebooks in [`notebooks/Data_preparation_subsets/`](notebooks/Data_preparation_subsets/) take the cleaned English corpus, split it into time periods, and apply deeper linguistic preprocessing to each period.

| Notebook | Purpose |
|---|---|
| `00_filter_and_split_by_period.ipynb` | Filters the corpus and splits it into the four periods. Run this first. |
| `01_preprocessing_1955_1970.ipynb` | Preprocessing for 1955–1970 |
| `02_preprocessing_1971_1986.ipynb` | Preprocessing for 1971–1986 |
| `03_preprocessing_1987_2002.ipynb` | Preprocessing for 1987–2002 |
| `04_preprocessing_2003_2018.ipynb` | Preprocessing for 2003–2018 |

**Filtering and splitting.** `00_filter_and_split_by_period.ipynb` removes documents with fewer than 20 tokens in `content_translated_processed` and keeps the years covered by the study (1955–2018). The remaining documents are grouped into consecutive 16-year windows based on the `year` column, and each window is saved as its own pickle file, named `data_<start>-<end>.pkl`. This gives the four periods analysed in the paper: 1955–1970, 1971–1986, 1987–2002 and 2003–2018.

**Preprocessing.** In notebooks `01` to `04`, each period is then prepared in two variants: a fully preprocessed one and a lightly cleaned one. The notebooks call them *word-level* and *sentence-level*. The names refer to how much the text is cleaned, not to the embedding model: in both cases BERTopic embeds whole documents with a sentence-embedding model.

| | Word-level variant | Sentence-level variant |
|---|---|---|
| Used for | LDA, NMF and all BERTopic results reported in the paper | Additional BERTopic experiments on lightly cleaned text |
| Regex cleaning | Removes repeated full stops, letters, symbols and phrases left by OCR | Same, and also removes all symbols except full stops |
| Stopwords | Removed (NLTK and spaCy) | Kept |
| Lemmatisation | NLTK `WordNetLemmatizer`, compared against spaCy | None |
| Punctuation | Removed | Full stops kept to preserve sentence boundaries |
| Short tokens | Words of one or two characters removed | Single-character tokens removed |
| Final column | `content_cleaned_spacy_lemmatized_cleaned` | `content_translated_processed_regex_cleaned` |
| Output file | `data_<period>_cleaned_df_regex_lemma_stop.pkl` | `data_<period>_sent_df_regex_cleaned.pkl` |

Long-running steps are applied in batches of 1,000 documents. An OCR spell-correction step with OCRFixr was tested and is left commented out in the notebooks.

**Input and output.** Notebook `00` reads `datasets/processed/ip_df_english_processed_full.pkl`, the output of the cleaning notebook, and all notebooks write to `datasets/data_split_timeframe/`. Paths are relative to the notebook folder; adjust them to your setup.

**Requirements:** `pandas`, `nltk` (with `punkt`, `stopwords`, `wordnet`), `spacy` with the `en_core_web_sm` model, and `swifter`.

## 🧩 Topic Modeling with BERTopic (Static and Dynamic)

The notebooks in [`notebooks/Topic_modeling/`](notebooks/Topic_modeling/) fit a BERTopic model on each time period and analyse how its topics develop over time. There is one notebook per period, `bertopic_static_dynamic_<period>.ipynb`, and all four follow the same steps:

1. **Embeddings.** Every document of the period is embedded with `Alibaba-NLP/gte-base-en-v1.5`.
2. **Static topic model.** BERTopic is fitted once on all documents of the period (UMAP, HDBSCAN, c-TF-IDF), with KeyBERT-inspired and MMR topic representations.
3. **Outlier reduction.** Documents that HDBSCAN leaves unassigned are assigned to their most probable topic.
4. **Dynamic topic modeling.** The fitted model is not retrained. Documents are grouped by publication year, and the frequency and representation of each topic are computed per year (`topics_over_time`).
5. **Visualisation.** Topics over time, topic word bar charts and the topic hierarchy are saved as interactive Plotly HTML files.

The input is the fully preprocessed text of each period from `notebooks/Data_preparation_subsets/`. Trained models and figures are written to `models/` and `Figures/` at the root of the repository.

> **Note:** The notebooks are published without outputs, because topic tables and representative documents contain text from the copyrighted corpus.

## 🎛️ BERTopic Hyperparameter Tuning

[`notebooks/Comparative_Study/Hyperparameter_tuning/hyperparameter_tuning_bertopic.ipynb`](notebooks/Comparative_Study/Hyperparameter_tuning/hyperparameter_tuning_bertopic.ipynb) tunes the UMAP and HDBSCAN settings of BERTopic with [Hyperopt](https://github.com/hyperopt/hyperopt) (Tree of Parzen Estimators), optimising topic coherence and topic diversity.

> **Note:** The tuning and evaluation code expects each dataset in [OCTIS](https://github.com/MIND-Lab/OCTIS) format: a folder containing `corpus.tsv` (one document per line, tab-separated, with the document text, its partition `train`, `val` or `test`, and an optional label) and `vocabulary.txt` (one word per line). For BERTopic, the document embeddings are computed beforehand and saved as a `.npy` file, with one row per document in the same order as the corpus that OCTIS loads.

> **Note:** The same procedure applies to any other embedding model or time period. Change the model name, the embeddings file and the dataset path in the settings cell at the top of the notebook.

## 📊 Evaluation Code

[`notebooks/evaluation/`](notebooks/evaluation/) contains the code used to train and score the topic models. It is imported by the hyperparameter-tuning and comparative-study notebooks.

| File | Purpose |
|---|---|
| `evaluation.py` | `Trainer`: trains a topic model and scores it on topic coherence (NPMI) and topic diversity, and records runtime and number of topics |
| `data.py` | `DataLoader`: loads a dataset in OCTIS format |
| `results.py` | `Results`: collects result files into comparison tables |
| `LDAtfidf.py`, `NMFtfidf.py` | LDA and NMF on TF-IDF weighted input |
| `LDA_NER.py`, `NMF_NER.py` | LDA and NMF on bag-of-words input augmented with named entities |
| `Top2Vec.py` | Customised Top2Vec |

This code is adapted from [BERTopic evaluation](https://github.com/MaartenGr/BERTopic_evaluation), [OCTIS](https://github.com/MIND-Lab/OCTIS) and [Top2Vec](https://github.com/ddangelov/Top2Vec).

## 🆚 Model Comparison

The notebooks in [`notebooks/Comparative_Study/model_comparison/`](notebooks/Comparative_Study/model_comparison/) compare BERTopic with the classical baselines on each time period. There is one folder per period (`1955_1970`, `1971_1986`, `1987_2002`, `2003_2018`):

| Notebook | Purpose |
|---|---|
| `Evaluation_script-topics-aggregation.ipynb` | Trains and scores LDA, NMF, their TF-IDF and named-entity variants, and BERTopic with default and tuned hyperparameters |
| `Evaluation_script_Top2Vec.ipynb` | Trains and scores Top2Vec as an additional baseline |
| `Exploratory_data_analysis.ipynb` | Exploratory analysis of the period: documents per year, vocabulary size and document length |

Every model is evaluated for 10, 20, 30, 40 and 50 topics on topic coherence (NPMI), topic diversity and runtime. LDA and NMF take the number of topics as a parameter. BERTopic finds its topics by clustering, and they are then merged to the same topic counts with `nr_topics`.

The notebooks read the datasets and precomputed embeddings described in the note above, and write their result files to `notebooks/evaluation_results/`, which is not part of this repository.

## 🔒 Data Access

Due to copyright restrictions, the dataset itself cannot be published. Access can be requested through the Impresso platform:

1. Register at https://impresso-project.ch/app/
2. Accept the terms of use
3. Sign the NDA: https://impresso-project.ch/assets/documents/impresso_NDA.pdf
4. Request access using the provided UIDs and keywords

The authors do not manage Impresso user accounts or dataset distribution, so some delay in access may occur.

## 📚 Citation

This repository accompanies the paper *Automating Historical Insight Extraction from Large-Scale Newspaper Archives using BERTopic*, which has been submitted to the journal *Digital Scholarship in the Humanities* (DSH) and is currently under peer review. The citation will be updated on publication.

```bibtex
@unpublished{murugaraj2025automating,
  title  = {Automating Historical Insight Extraction from Large-Scale Newspaper Archives using BERTopic},
  author = {Murugaraj, Keerthana and Lamsiyah, Salima and During, Marten and Theobald, Martin},
  note   = {Submitted to Digital Scholarship in the Humanities (DSH); under peer review},
  year   = {2025}
}
```

## 🙏 Acknowledgements

We thank the Impresso team for providing the document collection used in this study, based on their project *Impresso – Media Monitoring of the Past II*, funded by the Swiss National Science Foundation (SNSF 213585) and the Luxembourg National Research Fund (17498891).

This work was supported by the Doctoral Training Unit *Deep Data Science for History* (D4H), a collaboration between the Luxembourg Centre for Contemporary and Digital History (C²DH) and the Department of Computer Science (DCS), both at the University of Luxembourg.

## 📄 License

The code in this repository is released under the [MIT License](LICENSE). You are free to use, modify and redistribute it, including for other corpora and research questions, provided the copyright notice and license text are kept. If you use it in academic work, please cite the paper (see the Citation section).

The code in `notebooks/evaluation/` is adapted from third-party open-source projects; their licenses are reproduced in [`notebooks/evaluation/THIRD_PARTY_LICENSES`](notebooks/evaluation/THIRD_PARTY_LICENSES).

The license covers the code only. The newspaper corpus is not part of this repository and remains subject to the Impresso terms of use.
