# ASR for Afrikaans & isiXhosa [Code]
## Author: Lucas Meyer

Here we discuss our code. If you would like to use this code:
 - Install the required Python libraries using the ``requirements.txt`` file.
 - Replace "lucas-meyer" everywhere in the code with your own HuggingFace account name.
 - Enter your HuggingFace token as the string value of the variable ``WRITE_ACCESS_TOKEN`` in the ``utils.py`` file.

### 1. Data
We combined recordings from three data sources to create the following two datasets:
 - [Afrikaans ASR dataset (``asr_af``)](https://huggingface.co/datasets/lucas-meyer/asr_af)
 - [isiXhosa ASR dataset (``asr_xh``)](https://huggingface.co/datasets/lucas-meyer/asr_xh)

The following three data sources were used:
| Webpage Link | Language | Number of data entries | Additional info |
|--------------|----------|------------------------|-----------------|
|**NHCLT**||||
| [nchlt isixhosa](https://repo.sadilar.org/handle/20.500.12185/279)                 | xh      | 46651       | 107 different female speakers and 103 different male speakers. Xhosa speaker split: 103/99, 0/0, 4/4. There is also age information. |
| [nchlt afrikaans](https://repo.sadilar.org/handle/20.500.12185/280)                | af      | 66133       | 106 different female speakers and 103 different male speakers. Afrikaans speaker split: 102/99, 0/0, 4/4. There is also age information. |
|**High_Quality_TTS**||||
| [high quality tts](https://repo.sadilar.org/handle/20.500.12185/527)               | af + xh | 2927 + 2420 | Afrikaans: 9 different female speakers and 0 male speakers. Xhosa: 12 different female speakers and 0 male speakers. |
|**Fleurs_ASR**||||
| [hugging face fleurs](https://huggingface.co/datasets/google/fleurs)               | af + xh | 1494 + 4953 | Afrikaans amount split: 941/91, 0/198, 0/264. Xhosa amount split: 2471/995, 0/446, 0/1041. |

### 2. Loading data with Python
The following files were used to load data:
 - ``load_fleurs_nl.py``: Used to load the Dutch (Netherlands) speech data from the FLEURS dataset.
 - ``load_fleurs_zu.py``: Used to load the isiZulu speech data from the FLEURS dataset.
 - ``load_fleurs.py``: Used to load the Afrikaans and isiXhosa speech data from the FLEURS dataset.
 - ``load_high_quality_tts.py``: Used to load the Afrikaans and isiXhosa speech data from the High Quality TTS dataset.
 - ``load_nchlt.py``: Used to load the Afrikaans and isiXhosa speech data from the NCHLT dataset.

### 3. Training a model - ``fine_tune.ipynb``
We used a Python notebook (``fine_tune.ipynb``) to train our models. Typically, we would upload the notebook to a Google Colab session and use their available GPUs to run the code (takes about 5-6 hours to train a model). Note that all of the scripts used to load data and ``utils.py

### 4. Boosting performance with a LM - ``add_LM.ipynb``
We used a Python notebook (``add_LM.ipynb``) to add LMs to our models. We have already pre-processed the text data used to train our LMs and the code for that is in ``get_LM_data.ipynb``. The text data can be found in ``data/language_model_data/``.

### 5. Testing model performance - ``test.ipynb``
We used a Python notebook (``test.ipynb``) to test our models.