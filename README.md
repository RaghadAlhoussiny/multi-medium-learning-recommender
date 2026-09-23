# Multi-Medium Learning Content Recommender

CM3070 Final Project — BSc Computer Science, University of London
Template: CM3005 Data Science, Project Idea 1

A recommendation system that suggests both Coursera courses and Goodreads
educational books for the same topic. Existing platforms recommend only within
their own catalogue; this system encodes both catalogues into one shared
semantic space so a course and a book can be compared directly, even though the
two datasets share no users, no identifiers and no vocabulary.

## The system

- 336 Coursera courses, filtered from 623, with 1,019,660 reviews
- 593 Goodreads books, filtered from 10,000, with 217,744 ratings
- Four models: popularity baseline, collaborative filtering (SVD),
  content-based (Sentence-BERT), and a hybrid combining all three
- Streamlit web application with three pages

## Running the application

Requires Python 3.9 or later.

1. Download this repository (Code -> Download ZIP) and extract it

2. Open a terminal inside the extracted folder

3. Install the dependencies:

       pip install -r requirements.txt

4. Run the app:

       streamlit run app.py

   If `streamlit` is not recognised, use `python -m streamlit run app.py`.
   If you use Anaconda, run these commands in Anaconda Prompt.

6. The app opens in your browser. If it does not, use the URL printed in the
   terminal, usually http://localhost:8501

The first run downloads the Sentence-BERT model, about 90 MB, so it may take a
minute. Later runs start immediately.

## Notebooks

Each notebook is saved with its outputs, so they can be read without running
anything, including directly on GitHub by clicking any .ipynb file above.

| Notebook | Purpose |
|---|---|
| 01_data_curation | Filters both raw datasets to the learning scope |
| 02_prototype | Popularity baseline and collaborative filtering for courses |
| 03_content_based | TF-IDF content model for courses |
| 04_Books_Prototype | Popularity baseline and collaborative filtering for books |
| 05_semantic_embeddings | Sentence-BERT encoding and cross-medium pairing |
| 06_hybrid_and_coldstart | Hybrid model and the cold-start route |
| 07_evaluation | All four models on Precision@K, Recall@K and NDCG@K |
| 08_export_factors | Exports the SVD item factors used by the app |

Notebooks 01, 02 and 07 cannot be re-run from this repository, because the raw
Coursera review files are too large for GitHub. Their outputs are saved in the
notebooks and every file they produce is included.

## Datasets

Not included here due to size. Download from:

- Coursera 2021: https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera
- Goodbooks-10K: https://github.com/zygmuntz/goodbooks-10k

## Author

Raghad Mahmoud
