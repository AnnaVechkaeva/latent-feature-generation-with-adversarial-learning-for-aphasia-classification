# Latent Feature Generation with Adversarial Learning for Aphasia Classification

Code for the paper by Anna Vechkaeva and Günter Neumann (DFKI), published at LREC 2020.

The goal is to classify aphasia type (Anomic, Broca's, Conduction, or healthy Control) from
AphasiaBank transcripts when there is little data. Each utterance is turned into a matrix of
word2vec vectors. A conditional GAN / conditional Wasserstein GAN learns to generate new
utterance matrices for a given class, so the latent features are generated and no text is.
The generated matrices are added to the training data of an utterance-level classifier.
Each subject's label is then predicted from the predictions for their utterances.

> **About this code.** This is the research code from 2019–2020, published as it was
> used for the paper. Files were renamed and description notes were added, and notebook
> outputs were cleared. The code itself was not changed or refactored. Expect hard-coded
> file names, commented-out experiment variants, and notebook cells that were run out of order.

## Pipeline

Run the steps in order, from a folder that contains the data files each step reads and writes.

| Step | File | Paper | Original file name |
|---|---|---|---|
| 1. Clean the AphasiaBank CHAT transcripts, extract the WAB labels | `code/01_preprocess_aphasiabank.py` | Sec. 3.2 | `data_preprocessing.py` |
| 2. Patient-wise train/test split, word2vec 28×300 utterance matrices | `code/02_vectorize_and_split.ipynb` | Sec. 3.2–3.3, Table 2 | `DataVectors_preprocess.ipynb` |
| 3. Train the cGAN and conditional WGAN, run the utterance classifier experiments | `code/03_gan_experiments.ipynb` | Sec. 3.4.3, 3.5.1–3.5.2, Tables 4–5 | `Untitled3.ipynb` |
| 3b. Conditional WGAN as a standalone script | `code/03b_train_wcgan.py` | Sec. 3.4.3 | `W-cGAN.py` |
| 4. Generate synthetic utterance vectors with a trained generator | `code/04_generate_synthetic_data.py` | Sec. 3.5.2 | `generating_data.py` |
| 5. Train the utterance-level classifier | `code/05_train_utterance_classifier.py` | Sec. 3.4.1 | `CNN_classification.py` |
| 6. Subject-level evaluation with the Max Class model (MC-1 … MC-7) | `code/06_evaluate_max_class.py` | Sec. 3.5.3, Tables 4–7 | `2lv_eval.py` |
| 7. Two-part training split, supervised subject-level classifiers (NB, MNB, DT, RF, KNN, SVM) | `code/07_subject_level_supervised.ipynb` | Sec. 3.5.4, Tables 3, 8, 9 | `Untitled5.ipynb` |

Additional notebooks in `code/analysis/` (not needed to reproduce the results):

| File | What it does | Original file name |
|---|---|---|
| `dataset_stats.ipynb` | Average WAB score and age per group | `Dataset_stats.ipynb` |
| `utterance_stats.ipynb` | Number and length of utterances per group | `Untitled6.ipynb` |
| `mode_collapse.ipynb` | PCA / t-SNE plots for checking generated data for mode collapse | `mode_collapse.ipynb` |

Label ids used throughout: `0 = Anomic, 1 = Broca, 2 = Conduction, 3 = Control`.

## Data

The repository contains **no data**. AphasiaBank is available to registered members
through TalkBank: https://aphasia.talkbank.org/. To rerun the code you need:

* the English AphasiaBank transcripts (aphasia and control groups)
* the pretrained GoogleNews word2vec vectors (`GoogleNews-vectors-negative300.bin`),
  https://code.google.com/archive/p/word2vec/

The intermediate `.npy` files and trained models (`.h5`) are not included either.
Steps 2–7 recreate them.

## Environment

The experiments ran on Python 3.6 with Keras 2.2.4 and TensorFlow 1.x (the saved models
report `tf.keras` 2.1.6-tf). The code uses TF1-only APIs such as `tf.ConfigProto` and
`tf.Session`, so it will not run on TensorFlow 2 without changes. See `requirements.txt`.

## Known issues

These are in the original code and were deliberately left unchanged:

* In step 2 the test vector array starts with one duplicated row. The notebook therefore
  uses `test_images[1:]`, but `06_evaluate_max_class.py` and the evaluation cells in step 7
  index vectors and labels without the offset.
* The train/test split takes the first 30 subjects per class in `os.walk` order, so it
  depends on the file system order.
* The GAN generators and discriminators/critics are fully connected networks, and the paper
  describes them as CNNs. The WGAN keeps `num_classes = 10` and a sigmoid critic output from
  the MNIST example code it was adapted from.
* In Table 3 of the paper the Anomic and Broca's rows are swapped. In the data, Anomic has
  5,433 / 5,406 utterances and Broca's has 1,346 / 1,409.

## Citation

```
@inproceedings{vechkaeva2020latent,
  title={Latent Feature Generation with Adversarial Learning for Aphasia Classification},
  author={Vechkaeva, Anna and Neumann, G{\"u}nter},
  booktitle={LREC 2020 Language Resources and Evaluation Conference 11-16 May 2020},
  pages={88},
  year={2020}
}
```

## Acknowledgements

The GAN code is adapted from [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)
by Erik Linder-Norén (MIT License). This work was partially funded by the European Union's
Horizon 2020 grant agreement No. 777107 (Precise4Q) and by the BMBF project DeepLee (01IW17001).
