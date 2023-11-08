# ASR for Afrikaans & isiXhosa
## Author: Lucas Meyer
The focus of this project is performing ASR for Afrikaans and isiXhosa with limited speech data and computational resources. Our main research question is determining whether the use of additional Dutch and isiZulu speech data can help develop ASR systems for Afrikaans and isiXhosa. The report is given in the root directory of this repository.

This project is presented in partial fulfilment of the requirements for the degree of Master of Machine Learning and Artificial Intelligence in the Faculty of Applied Mathematics at Stellenbosch University. 

Supervisor: [Herman Kamper](https://www.kamperh.com/).

### 1. Code and trained models
The code used for this project is located in the ``src`` directory.
We train models by fine-tuning [XLS-R (300M)](https://huggingface.co/facebook/wav2vec2-xls-r-300m) and all of our trained models are stored on the [HuggingFace hub](https://huggingface.co/lucas-meyer?sort_models=alphabetical#models).

### 2. Related resources
 - (Docs) HuggingFace [datasets](https://huggingface.co/docs/datasets/index) and [transformers](https://huggingface.co/docs/transformers/index).
 - (Docs) HuggingFace [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) documentation.
 - (Blogpost) [XLS-R Fine-tuning tutorial](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)
 - (Blogpost) [Boosting Wav2Vec2 with n-grams](https://huggingface.co/blog/wav2vec2-with-ngram)
 - (Blogpost) [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
 - (YT video) [Transformer Neural Networks explained](https://www.youtube.com/watch?v=TQQlZhbC5ps)
 - (YT video) [GELU](https://www.youtube.com/watch?v=kMpptn-6jaw)
 - (YT video) [Layer normalization](https://youtube.com/shorts/TKPowx9fb-A?feature=share)
 - (YT video) [Grouped convolution](https://www.youtube.com/watch?v=3NU2vV3XD8c)
 - (YT video) [Self-supervised learning explained](https://www.youtube.com/watch?v=iGJ1XSkCyU0)
 - (YT video) [Beam search](https://www.youtube.com/watch?v=RLWuzLLSIgw) and [Refining beam search](https://www.youtube.com/watch?v=gb__z7LlN_4)
