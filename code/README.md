
### Preprocess data:

    #Generate training/valid/test dataframes.
    python preprocess_debates.py --input_path ../../data --output_path ../../data --topic_map_path ../../data/believe-based-topics.tsv  --user_min_nb_of_answers 3


    #Generate data with claims as targets...
    python argument_mining.py --data_frame_path ../../data/train_df.csv --cminer_path ../../data/arg_mining --output_path ../../data/train_with_claim_df.csv
    python argument_mining.py --data_frame_path ../../data/valid_df.csv --cminer_path ../../data/arg_mining --output_path ../../data/valid_with_claim_df.csv
    python argument_mining.py --data_frame_path ../../data/test_df.csv --cminer_path ../../data/arg_mining --output_path ../../data/test_with_claim_df.csv