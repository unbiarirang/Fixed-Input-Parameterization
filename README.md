# persona
This repository contains the official code for the paper: "Prompt Injection: Parameterization of Fixed Inputs"

## How to use

### PersonaChat

Data source
```
cd $DATA_DIR
wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json
```

Preprocessing
```
cd $REPO_DIR/scripts
python extract_persosna_chat_json.py $DATA_DIR/personachat_self_original.json train $OUTPUT_DIR/personachat_self_original_train.jsonl

python format_persona_chat_utterances.py $OUTPUT_DIR/personachat_self_original_train.jsonl $OUTPUT_DIR/personachat_self_original_train_formatted.jsonl  --permute --max-history 2 --tag
python format_persona_chat_utterances.py $DATA_DIR/personachat_benchmark_100.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl --max-history 2 --tag

python format_persona_chat_utterances_for_fid.py $OUTPUT_DIR/personachat_self_original_train.jsonl $OUTPUT_DIR/personachat_self_original_train_formatted_for_fid.jsonl
python format_persona_chat_utterances_for_fid.py $DATA_DIR/personachat_benchmark_100.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_for_fid.jsonl
```

Preprocessing (long prefix)
```
python make_multiple_persona.py $DATA_DIR/personas.csv $DATA_DIR/personachat_benchmark_100.jsonl $OUTPUT_DIR

python format_persona_chat_utterances.py $OUTPUT_DIR/benchmark_persona_chat_10personas.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_10personas.jsonl --max-history 2 --tag
python format_persona_chat_utterances.py $OUTPUT_DIR/benchmark_persona_chat_25personas.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_25personas.jsonl --max-history 2 --tag
python format_persona_chat_utterances.py $OUTPUT_DIR/benchmark_persona_chat_50personas.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_50personas.jsonl --max-history 2 --tag

python format_persona_chat_utterances_for_fid.py $OUTPUT_DIR/benchmark_persona_chat_10personas.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_10personas_for_fid.jsonl --long-prefix
python format_persona_chat_utterances_for_fid.py $OUTPUT_DIR/benchmark_persona_chat_25personas.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_25personas_for_fid.jsonl --long-prefix
python format_persona_chat_utterances_for_fid.py $OUTPUT_DIR/benchmark_persona_chat_50personas.jsonl $OUTPUT_DIR/benchmark_persona_chat_formatted_50personas_for_fid.jsonl --long-prefix
```

#### Lower-bound (student)
Note there's `--chat` flag for training models for personachat.
Low-bound model (student).
```
cd $REPO_DIR/scripts
python run_seq2seq_student.py \
  --model_name_or_path t5-base \
  --train_file $OUTPUT_DIR/personachat_self_original_train_formatted.jsonl \
  --validation_file $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --context_column context \
  --question_column input \
  --answer_column target \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --eval_accumulation_steps 20 \
  --predict_with_generate \
  --save_steps 5000 \
  --save_total_limit 1 \
  --chat \
  --output_dir $OUTPUT_DIR/$CHAT_STUDENT_DIR
```

Evaluating a student.
```
cd $REPO_DIR
python main.py \
  --method evaluate-only \
  --dataset personachat \
  --valid-path $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --model-path $OUTPUT_DIR/$CHAT_STUDENT_DIR \
  --chat \
  --name $RUN_NAME
```

#### Upper-bound (teacher)

Training a teacher:
```
cd $REPO_DIR/scripts
python run_seq2seq_teacher.py \
  --model_name_or_path t5-base \
  --train_file $OUTPUT_DIR/personachat_self_original_train_formatted.jsonl \
  --validation_file $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --context_column context \
  --question_column input \
  --answer_column target \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --eval_accumulation_steps 20 \
  --predict_with_generate \
  --save_steps 5000 \
  --save_total_limit 1 \
  --chat \
  --output_dir $OUTPUT_DIR/$CHAT_TEACHER_DIR
```

Evaluating a teacher:
```
cd $REPO_DIR
python main.py \
  --method evaluate-only \
  --dataset personachat \
  --with-prompt \
  --valid-path $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --model-path $OUTPUT_DIR/$CHAT_TEACHER_DIR \
  --chat \
  --name $RUN_NAME
```

#### Language modeling

Main training command (language modeling)
```
cd $REPO_DIR
python main.py \
  --method lm-finetune \
  --dataset personachat \
  --valid-path $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --model-path $OUTPUT_DIR/$CHAT_STUDENT_DIR \
  --num-steps 1000 \
  --valid-every-steps 100 \
  --print-valid-results \
  --chat \
  --name $RUN_NAME
```

#### Distillation (with input generation)

Training input-generator
```
cd $REPO_DIR/scripts
python run_seq2seq_input_generator.py \
  --model_name_or_path t5-base \
  --train_file $OUTPUT_DIR/personachat_self_original_train_formatted.jsonl \
  --validation_file $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --context_column context \
  --question_column input \
  --answer_column target \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --eval_accumulation_steps 20 \
  --predict_with_generate \
  --save_steps 5000 \
  --save_total_limit 1 \
  --chat \
  --output_dir $OUTPUT_DIR/$CHAT_INPUT_GENERATOR_DIR
```

Distillation command:
```
cd $REPO_DIR
python main.py \
  --method input-generation \
  --dataset personachat \
  --valid-path $OUTPUT_DIR/benchmark_persona_chat_formatted.jsonl \
  --model-path $OUTPUT_DIR/$CHAT_STUDENT_DIR \
  --teacher-path $OUTPUT_DIR/$CHAT_TEACHER_DIR \
  --input-generator-path $OUTPUT_DIR/$CHAT_INPUT_GENERATOR_DIR \
  --num-steps 1000 \
  --valid-every-steps 100 \
  --print-valid-results \
  --chat \
  --name $RUN_NAME
```
