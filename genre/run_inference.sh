python3 /home/users/nus/t0927864/scratch/nlp_hub/GENRE/genre/inference.py \
    --dataset_path /home/users/nus/t0927864/scratch/damuel/en/damuel_1.0_en/all_data_clean \
    --output_path /home/users/nus/t0927864/scratch/damuel/en/damuel_1.0_en/all_data_clean_mgenre_results \
    --model_path /home/users/nus/t0927864/scratch/nlp_hub/models/fairseq_multilingual_entity_disambiguation \
    --knowledge_base_path /home/users/nus/t0927864/scratch/damuel/en/damuel_1.0_en/knowledge_base.pickle \
    --candidate_marisa_trie_path /home/users/nus/t0927864/scratch/nlp_hub/data/mgenre/mgenre_marisa_trie \
    --batch_size 200