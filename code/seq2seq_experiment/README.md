
### Experiment:
- Run on users that have at least 3 answers...
- Text generation model generates one sentence that represent the claim

### Commands:

- Create opennmt files:
	

    python opennmt_data_preparation.py preprocess --data_path ../../data/train_with_claim_df.csv --output ../../data/opennmt_data/train --encoding_model_path path_to_glov_embedding/glove.42B.300d.txt
    python opennmt_data_preparation.py preprocess --data_path ../../data/valid_with_claim_df.csv --output ../../data/opennmt_data/valid --encoding_model_path path_to_glov_embedding/glove.42B.300d.txt
    python opennmt_data_preparation.py preprocess --data_path ../../data/test_with_claim_df.csv  --output ../../data/opennmt_data/test  --encoding_model_path path_to_glov_embedding/glove.42B.300d.txt


- Train a baseline:

	- Generate Vocab:
					

    #preprocess data claim as target
    python preprocess.py -train_src /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/train_with_claim.txt.src \
                             -train_tgt  /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/train_with_claim.txt.tgt \
                             -valid_src /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/valid_with_claim.txt.src \
                             -valid_tgt /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/valid_with_claim.txt.tgt \
                             -save_data  /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim \
                             -src_seq_length 10000 \
                             -tgt_seq_length 10000 \
                             -src_seq_length_trunc 100 \
                             -tgt_seq_length_trunc 100 \
                             -dynamic_dict \
                             -share_vocab \
                             -shard_size 100000
  
    - Generate Embedding:
		   

    ./tools/embeddings_to_torch.py -emb_file_enc /workspace/ceph_data/glove/glove.42B.300d.txt \
                                   -emb_file_dec /workspace/ceph_data/glove/glove.42B.300d.txt \
                                   -dict_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data.vocab.pt \
                                   -output_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data.vocab.pt.embeddings

    ./tools/embeddings_to_torch.py -emb_file_enc /workspace/ceph_data/glove/glove.42B.300d.txt \
                                   -emb_file_dec /workspace/ceph_data/glove/glove.42B.300d.txt \
                                   -dict_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_extended.vocab.pt \
                                   -output_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_extended.vocab.pt.embeddings



    ./tools/embeddings_to_torch.py -emb_file_enc /workspace/ceph_data/glove/glove.42B.300d.txt \
                                   -emb_file_dec /workspace/ceph_data/glove/glove.42B.300d.txt \
                                   -dict_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim.vocab.pt \
                                   -output_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim.vocab.pt.embeddings

    - Train:
	
    ##BASIC MODEL on claim as target:
    CUDA_VISIBLE_DEVICES=2 python train.py -data /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim \
                -path_to_train_profiles_feats /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/train_with_claim.users_embeddings.pkl \
                -path_to_valid_profiles_feats /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/valid_with_claim.users_embeddings.pkl \
                -path_to_train_key_phrases /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/train_with_claim.topic_key_phrases.pkl \
                -path_to_valid_key_phrases /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/valid_with_claim.topic_key_phrases.pkl \
                -save_model /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/models/basic_claim \
                -pre_word_vecs_enc  /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim.vocab.pt.embeddings.enc.pt \
                -pre_word_vecs_dec  /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim.vocab.pt.embeddings.dec.pt \
                -log_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/logs/basic_claim.log \
                -keep_checkpoint 5 \
                -save_checkpoint_steps 2000 \
                -valid_steps 2000 \
                -train_steps 90000 \
                -enc_rnn_size 512 \
                -dec_rnn_size 512 \
                -layers 2 \
                -word_vec_size 300 \
                -multimodal_model_type other \
                -global_attention mlp \
                -encoder_type rnn \
                -decoder_type rnn \
                -input_feed 1 \
                -max_grad_norm 2 \
                -dropout 0.5 \
                -batch_size 16 \
                -valid_batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -seed 777\
                -world_size 1 \
                -gpu_ranks 0


    #Approach on claim as target on big-issues
    CUDA_VISIBLE_DEVICES=3 python train.py -data /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim \
                        -path_to_train_profiles_feats /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/train_with_claim.users_big_issues.pkl \
                        -path_to_valid_profiles_feats /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/valid_with_claim.users_big_issues.pkl \
                        -path_to_train_key_phrases /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/train_with_claim.topic_key_phrases.pkl \
                        -path_to_valid_key_phrases /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/valid_with_claim.topic_key_phrases.pkl \
                        -save_model /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/models/persona_based \
                        -pre_word_vecs_enc  /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim.vocab.pt.embeddings.enc.pt \
                        -pre_word_vecs_dec  /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/opennmt_data/data_with_claim.vocab.pt.embeddings.dec.pt \
                        -log_file /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/logs/persona_based.log \
                        -keep_checkpoint 5 \
                        -save_checkpoint_steps 2000 \
                        -valid_steps 2000 \
                        -train_steps 120000 \
                        -enc_rnn_size 512 \
                        -dec_rnn_size 512 \
                        -layers 2\
                        -word_vec_size 300 \
                        -project_user \
                        -user_feat_size 192 \
                        -user_hidden_size 16 \
                        -key_phrases_feat_size 300 \
                        -project_key_phrases \
                        -use_nonlinear_projection \
                        -context_dropout 0.5\
                        -multimodal_model_type doubly-attn \
                        -global_attention mlp \
                        -encoder_type rnn \
                        -decoder_type doubly-attn \
                        -input_feed 1 \
                        -max_grad_norm 2 \
                        -dropout 0.5 \
                        -batch_size 16 \
                        -valid_batch_size 16 \
                        -optim adagrad \
                        -learning_rate 0.15 \
                        -adagrad_accumulator_init 0.1 \
                        -seed 777\
                        -world_size 1 \
                        -gpu_ranks 0


- Inference:
        
        python generate_preds.py --seq2seq_basic_model_path /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/models/basic_claim_step_120000.pt \
                                        --seq2seq_app_model_path /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/models/persona_based_step_120000.pt \
                                         --training_dataset_path /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/preprocessed_data/train_with_claim_df.csv \
                                         --valid_dataset_path   /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/preprocessed_data/test_with_claim_df.csv \
                                         --users_df_path        /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/embedded_data/autoencoder_train3000.parquet.gzip \
                                         --output_path          /workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/output/persona_based_test_preds.csv