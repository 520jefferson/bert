export CUDA_VISIBLE_DEVICES="3"
export BERT_BASE_DIR=chinese_roberta_wwm_ext_L-12_H-768_A-12
export GLUE_DIR=data

python run_classifier.py \
  --task_name=COLA \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/CoLA_movie_2class \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=196 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=run_train_movies/
