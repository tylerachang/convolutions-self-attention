# convolutions-and-self-attention
Experiments for integrating convolutions and self-attention.
Code adapted from https://github.com/huggingface/transformers.
Run on Python 3.6.9.

## Training
To train tokenizer, use custom_scripts/train_spm_tokenizer.py.
To pre-train BERT with a plain text dataset:
<pre>
python3 run_language_modeling.py \
--model_type=bert \
--tokenizer_name="./data/sentencepiece/spm.model" \
--config_name="./data/bert_base_config.json" \
--do_train --mlm --line_by_line \
--train_data_file="./data/training_text.txt" \
--per_device_train_batch_size=32 \
--save_steps=25000 \
--block_size=128 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--output_dir="./bert-experiments/bert"
</pre>

The code above produces a cached file of examples (a list of lists of token indices).
Each example is an un-truncated and un-padded sentence pair (but includes [CLS] and [SEP] tokens).
Convert these lists to an iterable text file using custom_scripts/shuffle_cached_dataset.py.
Then, you can pre-train BERT using an iterable dataset (saving memory):
<pre>
python3 run_language_modeling.py \
--model_type=bert \
--tokenizer_name="./data/sentencepiece/spm.model" \
--config_name="./data/bert_base_config.json" \
--do_train --mlm --train_iterable --line_by_line \
--train_data_file="./data/iterable_pairs_train.txt" \
--per_device_train_batch_size=32 \
--save_steps=25000 \
--block_size=128 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--output_dir="./bert-experiments/bert"
</pre>

Optional flags to change BERT architecture when pre-training from scratch:<br/>
In the following, qk uses query/key attention, convfixed is a fixed lightweight convolution, convq is query-based dynamic lightweight convolution (relative embeddings), convk is a key-based dynamic lightweight convolution, and convolution is a fixed depthwise convolution.
<pre>--attention_kernel="[qk]_[convfixed]_[convq]_[convk] [num_positions_each_dir]"</pre>
Remove absolute position embeddings:
<pre>--remove_position_embeddings</pre>
Convolutional values for half of heads:
<pre>--value_forward="convolution_[depth]_[mixed]_[no_act] [num_positions_each_dir] [num_groups]"</pre>
Convolutional queries/keys for half of heads:
<pre>--qk="convolution_[depth]_[mixed]_[no_act] [num_positions_each_dir] [num_groups]"</pre>

## Fine-tuning
Training and evaluation for downstream GLUE tasks (note: batch size represents max batch size, because batch size is adjusted for each task):
<pre>
python3 run_glue.py \
--data_dir="./glue-data/data-tsv" \
--task_name=ALL \
--save_steps=9999999 \
--max_seq_length 128 \
--per_device_train_batch_size 99999 \
--tokenizer_name="./data/sentencepiece/spm.model" \
--model_name_or_path="./bert-experiments/bert" \
--output_dir="./bert-experiments/bert-glue" \
--hyperparams="electra_base" \
--do_eval \
--do_train
</pre>

## Prediction
Run the fine-tuned models on the GLUE test set:<br/>
This adds a file with test set predictions to each GLUE task directory.
<pre>
python3 run_glue.py \
--data_dir="./glue-data/data-tsv" \
--task_name=ALL \
--save_steps=9999999 \
--max_seq_length 128 \
--per_device_train_batch_size 99999 \
--tokenizer_name="./data/sentencepiece/spm.model" \
--model_name_or_path="./bert-experiments/placeholder" \
--output_dir="./bert-experiments/bert-glue" \
--hyperparams="electra_base" \
--do_predict
</pre>
Then, test results can be compiled into one directory.
The test_results directory will contain test predictions, using the finetuned model with the highest dev set score for each task.
The files in test_results can be zipped and submitted to the GLUE benchmark site for evaluation.
<pre>
python3 custom_scripts/parse_glue.py \
--input="./bert-experiments/bert-glue" \
--test_dir="./bert-experiments/bert-glue/test_results"
</pre>
