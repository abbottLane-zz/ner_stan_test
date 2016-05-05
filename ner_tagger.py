from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

#######################################################
### TEST NER TAGGER ON A SIMPLE INPUT #################
#######################################################

stanford_tagger = StanfordNERTagger(
    'Resources/NER_Models/english.muc.7class.nodistsim.crf.ser.gz',
    '/home/wlane/stanford-ner-2015-04-20/stanford-ner.jar',
    encoding='utf-8')

text = 'While surfing in Morocco on June 24th, Jane figured out that Danny DeVito was pretending to be a shark and ' \
       'mauling tourists in the water at Exxon Corp.\'s Bayside Resort'
tokenized_text = word_tokenize(text)
classified_text = stanford_tagger.tag(tokenized_text)

print(classified_text)

#########################################################
## TRAIN our own NER model : JaneAusten "Emma" Tutorial #
#########################################################

# # Tokenize the training text
# with open("Resources/emmaCh1.txt") as f:
#     data=f.read().replace('\n', ' ')
# tokenized_emma = word_tokenize(data)

# Write output in format suitable for NER training input
# emma_train_file = open("Resources/emma_train.tsv", "w")
# for tok in tokenized_emma:
#     emma_train_file.write(tok + "\tO\n" ) # Ideally we'd mark entities as well
# emma_train_file.close()