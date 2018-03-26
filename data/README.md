### Processed data files

Since the limit of github, only one original data file is added. Since the data processing might take time and the use 
of nltk module can be unwanted.
The data files are uploaded to [google drive](https://drive.google.com/drive/folders/0Bwz-gNlFG7V8Nnp1bERjUG9TcEE?usp=sharing).
All data files can be read using pickle:

* Digital_Music_5.json is the raw data
* clean_data_wordlist, each item is in the format of [[list of words], label]
* clean_data_string, each item is in the format of [string of clean words, label ]
* clean_data_using_stopowrds_punkt as explained in the name
* clean_data_no_stopwords_punkt as explained in the name