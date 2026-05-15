# Neural-Machine-Translation
Building a Machine-Translation Transformer Model, That Translates from English to Arabic.  
> Final Project for `Pattern Recognition` Faculty course.

## Methodology
- This architecture focuses on building a Transformer model that translates relatively short conversational sentences.  
- To achieve this, I have used a combination of datasets
  - the conversational data `Helsinki-NLP/opus-100`
  - and `news_commentary` which has more official tone to help the model learn the language structure.
  - the third custom dataset is regular sentences.
- This model is also aimed to be light weight, so I have used data size relative to the scale of the model parameters count.
  - Limiting the number of training sentence pairs to `120k` pairs.
  - Limiting the sentences length to `30` words and discarding sentences having exceeding this limit.

## Text preprocessing
> In Text preprocessing for translation, removing stop words and punctuation prevents the model from being able to translate the stop words and the punctuation correctly. so i havne't messed with the stop words and punctuation and preserved them.
- ### For `Arabic & English` sentences:
    - (1) remove special characters
    - (2) remove links or HTML tags (that might exist in the datasets)
    - (3) replace any number of concatenated hyphens `-` by only one hyphen.
    - (4) replace any number of concatenated dots `.` by only one dot.
        - to ease the job on the tokenizer and prevent wasting vocab that contains multiple hyphens.
    - (5) replace any multiple spaces or tabs by only one space.
- ### For `Arabic` sentences only:
   - (1) remove Diacratics (Tashkeel) and Tatweel
   - (2) for characters that has multiple variants, they are replaced by the default variant (demonestrated in the notebook)
   - (3) removing quotes
   - (4) replace arabic comma and question marks with englsih one (to prevent consumption of limited vocab words by words having these character variants)
   - (5) replace arabic numbers with english ones.

## Used Datasets
1) `Helsinki-NLP/opus-100` `ar-en`
2) `news_commentary` `ar-en`
3) `Custom dataset` available in this [link](https://raw.githubusercontent.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/master/ara_.txt)

## Experiments logging
using `MLFLow` and `DagsHub` for logging different experiments on this link: [Experiments](https://dagshub.com/michael-nabil/Neural-Machine-Translation.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)
