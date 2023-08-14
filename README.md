# Speech Recognition for spoken Afrikaans and isiXhosa

The basic aim of this project is to recognize spoken numbers for math games 
in Afrikaans and isiXhosa. The end-goal of the project is to perform 
general speech recognition for primary educational games in Afrikaans and 
isiXhosa.

## How to run

Read through the ``src/main.ipynb`` notebook.

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
        - **Answer**: Choose test/validation set very carefully & don't split randomly.
   5. **Question:** Should I remove the English sentences from the Afrikaans and
      isiXhosa datasets?
        - **Answer:** No.
   6. **Question:** Siswati or isiXhosa?
        - **Answer:** Use isiXhosa for now.

## Progress (18/08/2023)

## Questions for Herman (18/08/2023)

   1. **Question:** What ... ?
        - **Answer:** ...

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

