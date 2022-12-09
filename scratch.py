# import re

# with open("1_PCA.txt", 'r') as file:
#     i = str(file.read())

# regex = re.compile(r'[^a-zA-Z\s]')
# i = regex.sub('', i)
# i = i.replace('\n', '')

# with open("1_PCA2.txt", "w") as f:
#     f.write(i)
# import coqui_stt_training
# from coqui_stt_training.train import train


initialize_globals_from_args(
    n_hidden=2048,
    # using the same save and load allows us to run test in teh same go.  
    load_checkpoint_dir="models/coqui/checkpoints/coqui-stt-1.0.0-checkpoint",
    save_checkpoint_dir="models/coqui/checkpoints/coqui-stt-1.0.0-checkpoint",
    # save_checkpoint_dir="models/coqui/checkpoints/1380",
    drop_source_layers=1,
    # alphabet_config_path="models/coqui/huge-vocabulary/alphabet.txt",
    epochs=5,
    load_cudnn=True,
    # train_cudnn=True,
    auto_input_dataset="models/coqui/1_PCAwav/coqui_train_PCA_1_0_1380"
)



docker run  -it \
  --entrypoint /bin/bash \
  --name stt-train \
  --gpus all \
  --mount type=bind,source="$(pwd)"/stt-data,target=/code/stt-data \
  ghcr.io/coqui-ai/stt-train:v0.10.0-alpha.4

  
python3 train.py \
  --train_files stt-data/models/coqui/PCAwav/train.csv \
  --dev_files stt-data/models/coqui/PCAwav/dev.csv \
  --test_files stt-data/models/coqui/PCAwav/test.csv \
  --checkpoint_dir stt-data/models/coqui/checkpoints/coqui-stt-1.0.0-checkpoint \
  --export_dir stt-data/coqui/docker/exported-model \
  --alphabet_config_path stt-data/models/coqui/PCAwav/alphabet.txt \
  --n_hidden 2048 \
  --drop_source_layers 1 \
  --epochs 100 \
  --train_cudnn True

# using original 
  python3 train.py \
  --train_files stt-data/models/coqui/PCAwav/train.csv \
  --dev_files stt-data/models/coqui/PCAwav/dev.csv \
  --test_files stt-data/models/coqui/PCAwav/test.csv \
  --checkpoint_dir stt-data/models/coqui/checkpoints/original2/coqui-stt-1.0.0-checkpoint \
  --export_dir stt-data/coqui/docker/exported-model \
  --alphabet_config_path stt-data/models/coqui/checkpoints/original2/coqui-stt-1.0.0-checkpoint/alphabet.txt \
  --n_hidden 2048 \
  --drop_source_layers 2 \
  --epochs 100 \
  --train_cudnn True \
  --early_stop True 


python3 train.py \
    --test_files stt-data/models/coqui/PCAwav/test.csv \
    --checkpoint_dir stt-data/models/coqui/checkpoints/original/coqui-stt-1.0.0-checkpoint


# kenLM - run this from the docker instance of coqui to create a trie file
python data/lm/generate_lm.py \
  --input_txt stt-data/1_PCA.mp4_sentences.txt \
  --output_dir stt-data/ken-lm-1/ \
  --top_k 500000 \
  --kenlm_bins kenlm/build/bin/ \
  --arpa_order 5 \
  --max_arpa_memory "85%" \
  --arpa_prune "0|0|1" \
  --binary_a_bits 255 \
  --binary_q_bits 8 \
  --binary_type trie

# checkpoint file
https://github.com/coqui-ai/STT/releases 
#   getting the kenlm scorer generator
curl -L https://github.com/coqui-ai/STT/releases/download/v1.4.0/native_client.tflite.Linux.tar
.xz | tar -Jxvf -

# running the scorer
./generate_scorer_package \
  --checkpoint stt-data/model/coqui-stt-1.4.0-checkpoint \
  --lm stt-data/ken-lm-1/lm.binary \
  --vocab stt-data/ken-lm-1/vocab-500000.txt \
  --package kenlm-pca-scorer-1 \
  --default_alpha 0.931289039105002 \
  --default_beta 1.1834137581510284
