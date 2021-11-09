#!/bin/bash
train='altaTrainTruth.npy'
test='altaTestTruth.npy'

#ALBERT
# python autoClassify.py train_albert_base.npy $train test_albert_base.npy $test
# python autoClassify.py train_albert_large.npy $train test_albert_large.npy $test
# python autoClassify.py train_albert_xlarge.npy $train test_albert_xlarge.npy $test

#XLNET

#  python autoClassify.py train_xlnet_large_cased.npy $train test_xlnet_large_cased.npy $test

#BERT
#  python autoClassify.py train_bert_base_cased.npy $train test_bert_base_cased.npy $test
#  python autoClassify.py train_bert_base_uncased.npy $train test_bert_base_uncased.npy $test
 python autoClassify.py train_bert_large_cased.npy $train test_bert_large_cased.npy $test
 python autoClassify.py train_bert_large_uncased.npy $train test_bert_large_uncased.npy $test

#ELMO
#  python autoClassify.py train_elmo_bi_lm.npy $train test_elmo_bi_lm.npy $test

#ULMFIT
#  python autoClassify.py train_ulmfit_forward.npy $train test_ulmfit_forward.npy $test
#  python autoClassify.py train_ulmfit_backward.npy $train test_ulmfit_backward.npy $test

#USE
#  python autoClassify.py train_use_elm.npy $train test_use_elm.npy $test
#  python autoClassify.py train_use_transformer_large.npy $train test_use_transformer_large.npy $test

#WordtestVec
#  python autoClassify.py train_google_news_300.npy $train test_google_news_300.npy $test

#fasttext 
#  python autoClassify.py train_common_crawl_300.npy $train test_common_crawl_300.npy $test
#  python autoClassify.py train_common_wiki_news_300.npy $train train_common_wiki_news_300.npy $test

#glove
python autoClassify.py train_twitter_200.npy $train test_twitter_200.npy $test
# python autoClassify.py train_wiki_300.npy $train test_wiki_300.npy $test
 python autoClassify.py train_crawl_42B_300.npy $train test_crawl_42B_300.npy $test