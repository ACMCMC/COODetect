# COODetect

Companion code for the article "COODetect at SemEval 2026 Task 13: Unsupervised Latent Domain Adaptation for Out-of-Distribution AI Code Detection"

Authors: Aldan Creo, Atharv Nair, Mohana Kannan Ravikumar, Vaishak Menon, Dario Wisznewer and Vaibhav Jain

## Running the code

From the project root, install dependencies first with `pip install -r requirements.txt`, verify that the parquet input paths in pipeline.yaml are correct, then run the feature extractor script at `1_extract_features.py`; `2_select_features.py`, and so on; if you want a quick test use `SAMPLE_N_ROWS=1000` before running, and if you want to ignore existing cached outputs use `FORCE_REBUILD=1`, and the generated feature arrays/metadata will be saved into cache.


