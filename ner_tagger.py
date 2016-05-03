from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

stanford_tagger = StanfordNERTagger('/home/wlane/compling/stanford-ner-2014-06-16/classifiers/english.muc.7class.nodistsim.crf.ser.gz',
					   '/home/wlane/compling/stanford-ner-2014-06-16/stanford-ner.jar',
					   encoding='utf-8')

text = 'While surfing in Morocco on June 24th, Jane figured out that Danny DeVito was pretending to be a shark and mauling tourists in the water at Exxon Corp.\'s Bayside Resort'
tokenized_text = word_tokenize(text)
classified_text = stanford_tagger.tag(tokenized_text)

print(classified_text)

