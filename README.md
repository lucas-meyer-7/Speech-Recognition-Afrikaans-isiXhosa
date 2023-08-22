# Speech Recognition for spoken Afrikaans and isiXhosa
The basic aim of this project is to recognize spoken numbers for math games 
in Afrikaans and isiXhosa. The end-goal of the project is to perform 
general speech recognition for primary educational games in Afrikaans and 
isiXhosa.

## How to run
Read through the ``src/main.ipynb`` notebook.

## Datasets (Work in progress)
| Webpage Link | Language | Number of data entries | Additional info |
|--------------|----------|------------------------|-----------------|
|**isiXhosa datasets**||||
| [nchlt isixhosa](https://repo.sadilar.org/handle/20.500.12185/279)                 | xh      | 46651       | 107 different female speakers and 103 different male speakers. Xhosa speaker split: 103/99, 0/0, 4/4. There is also age information. |
| [african speech tech](https://repo.sadilar.org/handle/20.500.12185/305)            | xh      | Not sure.   | Struggling with the format of the dataset. The audio files are in ``.alaw`` format and the full sentence transcriptions are not given (they provide ``.TextGrid`` files). |
| [lwazi 1 isixhosa tts](https://repo.sadilar.org/handle/20.500.12185/462)           | xh      | 339         | Nothing besides allignment info. |
| [lwazi 2 isixhosa tts](https://repo.sadilar.org/handle/20.500.12185/440)           | xh      | 912         | Nothing besides allignment info. |
| [lwazi 3 isixhosa tts **](https://repo.sadilar.org/handle/20.500.12185/268)        | xh      | Requested.  | Requested. |
| [lwazi isixhosa asr](https://repo.sadilar.org/handle/20.500.12185/460)             | xh      | 6300        | 210 different speakers (30 sentences each): 107 female speakers and 101 male speakers (2 unknown). The examples for each speaker are not very long, and the transcritptions need to be preprocessed in their own way. The spoken sentences are the same for every speaker. There is also age information. Cellphone quality recordings. |
|**Afrikaans datasets**||||
| [nchlt afrikaans](https://repo.sadilar.org/handle/20.500.12185/280)                | af      | 66133       | 106 different female speakers and 103 different male speakers. Afrikaans speaker split: 102/99, 0/0, 4/4. There is also age information. |
| [african speech tech](https://repo.sadilar.org/handle/20.500.12185/400)            | af      | Not sure.   | Struggling with the format of the dataset. The audio files are in ``.alaw`` format and the full sentence transcriptions are not given (they provide ``.TextGrid`` files). |
| [lwazi 1 afrikaans tts **](https://repo.sadilar.org/handle/20.500.12185/436)       | af      | Requested.  | Requested. |
| [lwazi 2 afrikaans tts **](https://repo.sadilar.org/handle/20.500.12185/443)       | af      | Requested.  | Requested. |
| [lwazi 3 afrikaans tts **](https://repo.sadilar.org/handle/20.500.12185/266)       | af      | Requested.  | Requested. |
| [lwazi afrikaans asr](https://repo.sadilar.org/handle/20.500.12185/434)            | af      | 6000        | 200 different speakers (30 sentences each): 98 female speakers and 101 male speakers (1 unknown). The examples for each speaker are not very long, and the transcritptions need to be preprocessed in their own way. The spoken sentences are the same for every speaker. There is also age information. Cellphone quality recordings. |
|**Weird Afrikaans datasets**||||
| [tracking trajectory](https://repo.sadilar.org/handle/20.500.12185/442)            | af      | 11097       | 1 male speaker. Short and mostly half sentences (+- three words). |
| [coloured afrikaans](https://repo.sadilar.org/handle/20.500.12185/444)             | af      | Not sure.   | Struggling with the format of the dataset. The audio files are in ``.alaw`` format and the full sentence transcriptions are not given (they provide ``.TextGrid`` files). |
| [black afrikaans](https://repo.sadilar.org/handle/20.500.12185/422)                | af      | Not sure.   | Struggling with the format of the dataset. The audio files are in ``.alaw`` format and the full sentence transcriptions are not given (they provide ``.TextGrid`` files). |
| [multipron corpus](https://repo.sadilar.org/handle/20.500.12185/379)               | af      | ~2500-3000  | Proper names dataset. 7 female speaker and 6 male speakers. |
|**Afrikaans + isiXhosa datasets**||||
| [high quality tts](https://repo.sadilar.org/handle/20.500.12185/527)               | af + xh | 2927 + 2420 | Afrikaans: 9 different female speakers and 0 male speakers. Xhosa: 12 different female speakers and 0 male speakers. |
| [hugging face fleurs](https://huggingface.co/datasets/google/fleurs)               | af + xh | 1494 + 4953 | Afrikaans amount split: 941/91, 0/198, 0/264. Xhosa amount split: 2471/995, 0/446, 0/1041. |

## Progress (11/08/2023)
   1. Downloaded Afrikaans and isiXhosa TTS datasets.
   2. Performed preprocessing to convert data into Dataset format for huggingface.
   3. Created a train/validation/test set from the data.
   4. Bare minimum text normalization for the label/sentences/transcripts.
   5. Imported large XLS-R model and fine-tuned with Afrikaans data + saved a model.

## Questions for Herman (11/08/2023)
   1. **Question:** What's next? In what direction are we heading towards?
        - **Answer:** First clean up pipeline and sort out datasets.
   2. **Question:** Do you expect me to implement and train a similar model from scratch?
        - **Answer:** ...
   3. **Question:** Should diacritics be removed? Other preprocessing tips?
        - **Answer:** No, keep it as simple as possible.
   4. **Question:** I'm not entirely sure what a sensible method is for 
      choosing a validation/test set. Right now I am splitting
      randomly.
        - **Answer:** Choose test/validation set very carefully & don't split randomly.
   5. **Question:** Should I remove the English sentences from the Afrikaans and
      isiXhosa datasets?
        - **Answer:** No.
   6. **Question:** Siswati or isiXhosa?
        - **Answer:** Use isiXhosa for now.

## Progress (18/08/2023)
   1. Downloaded more datasets.
   2. Familiarized with HuggingFace hub and can now load and store models.
   3. Further research into how XLS-R works.
   4. Research into how OpenAI's whisper works.

## Questions for Herman (18/08/2023)
   1. **Question:** *(More for myself)* Why are NNs preferred over other 
   machine learning algorithms for the problem of speech recognition?
        - **ChatGPT answer:** In summary, neural networks are preferred for 
        automatic speech recognition due to their capacity for automatic 
        feature learning, end-to-end modeling, robustness to variability, 
        utilization of large-scale data, and adaptability through transfer 
        learning. These characteristics collectively contribute to their 
        superior performance in ASR tasks.
        - **Herman answer:** Over the years NNs started to perform much better
        than the goto models, which were HMMs at the time.
   2. **Question:** Is the development set the same as the validation set? 
   Why do people use less data entries in their dev sets than in their test sets?
   	- **Answer:** Could be for many different reasons, but most likely it is
   	small so that the training set can be as large as possible.



## TODO (25/08/2023)
   1. Read through papers and make summaries.
   2. Load datasets and finish scripts.
   3. Start writing report.

## Progress (25/08/2023)
   1. 

## Questions for Herman (18/08/2023)
   1. **Question:**



## References and Acknowledgements
This following is a summary of the internet resources that were used
to develop this project.

### Papers (read and summarized):
 - Attention is all you need.
 
### Papers (to finish)
 - XLS-R
 - ProphetNet (kinda)
 - Wav2Vec
 - Whisper
 - Fourier series approximation paper from YT

### ChatGPT chats:
 - https://chat.openai.com/share/af4e27d1-3899-45b2-8917-bf3b5f6e669a
 - https://chat.openai.com/share/9da5e4fb-c90c-4050-aa62-045137ee0076
 - https://chat.openai.com/share/4885c5b0-1465-4cc9-a74b-a1282bf8ccc1

### Tutorial/blog webpages:
 - Fine-tuning tutorial: https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

### YouTube videos:
 - Transformer Neural Networks explained: https://www.youtube.com/watch?v=TQQlZhbC5ps 
 - HuggingFace introduction: https://www.youtube.com/watch?v=QEaBAZQCtwE

