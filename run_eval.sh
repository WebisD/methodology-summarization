python run_summarization.py \
--mode=eval \
--data_path=./data/sample250-test.bin \
--vocab_path=./data/sample250-test.vocab \
--log_root=logroot \
--exp_name=test-experiment \
--max_dec_steps=200 \
--max_enc_steps=2000 \
--num_sections=5 \
--max_section_len=400 \
--batch_size=4 \
--vocab_size=50000 \
--use_do=True \
--optimizer=adagrad \
--do_prob=0.25 \
--hier=True \
--split_intro=True \
--fixed_attn=True \
--legacy_encoder=False \
--coverage=False
