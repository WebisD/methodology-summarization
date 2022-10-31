python ../scripts/json_to_bin.py ./train.txt ./train.bin --vocab_file ./train.vocab && \
python ../scripts/json_to_bin.py ./test.txt ./test.bin --vocab_file ./test.vocab && \
python ../scripts/json_to_bin.py ./eval.txt ./eval.bin --vocab_file ./eval.vocab