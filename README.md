# Speech Recognition for spoken Afrikaans and isiXhosa

The basic aim of this project is to recognize spoken numbers for math games 
in Afrikaans and isiXhosa/Siswati. The end-goal of the project is to perform 
general speech recognition for primary educational games in Afrikaans and 
isiXhosa/Siswati.

## Progress

Completed the pipeline for fine tuning a hugging face model, which includes:
   
   1. Downloaded Afrikaans and isiXhosa TTS datasets.
   2. Performed preprocessing to convert data into Dataset format for huggingface.
   3. Created a train/validation/test set from the data.
   4. Bare minimum text normalization for the label/sentences/transcripts.
   5. Imported large XLS-R model and fine-tuned with Afrikaans data + saved a model.

## Questions (for Herman)

My main two questions are :

   1. What's next? In what direction are we heading towards?
   2. Do you expect me to implement and train a similar model from scratch?

My other questions are:

   1. Should diacritics be removed? Other preprocessing tips?
   2. I'm not entirely sure what a sensible method is for 
      choosing a validation/test set. Right now I am splitting randomly.
   3. Should I remove the English sentences from the Afrikaans/isiXhosa datasets?
   4. Siswati or isiXhosa?

## How to run (WIP)

Refer to notebook ``src/main.ipynb``, which contains all of the code that is
used to fine-tune XLS-R with the Afrikaans dataset.

## References and Acknowledgements

This following is a summary of the internet resources that were used
to develop this project.

### ChatGPT chats:

 - https://chat.openai.com/share/af4e27d1-3899-45b2-8917-bf3b5f6e669a
 - https://chat.openai.com/share/9da5e4fb-c90c-4050-aa62-045137ee0076

### Tutorial/blog webpages:

 - Fine-tuning tutorial: https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

### YouTube videos:

 - HuggingFace introduction: https://www.youtube.com/watch?v=QEaBAZQCtwE
