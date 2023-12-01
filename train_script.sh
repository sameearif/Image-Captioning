# # python data_generator.py \
# #     --save_model_path ./github_ignore_material/raw_data/swin_large_patch4_window12_384_22k.pth \
# #     --output_path ./github_ignore_material/raw_data/features.hdf5 \
# #     --images_path ./github_ignore_material/raw_data/ \
# #     --captions_path ./github_ignore_material/raw_data/


# python train.py --N_enc 3 --N_dec 3  \
#     --model_dim 512 --seed 775533 --optim_type radam --sched_type custom_warmup_anneal  \
#     --warmup 10000 --lr 2e-4 --anneal_coeff 0.8 --anneal_every_epoch 2 --enc_drop 0.3 \
#     --dec_drop 0.3 --enc_input_drop 0.3 --dec_input_drop 0.3 --drop_other 0.3  \
#     --batch_size 16 --num_accum 1 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
#     --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 3  \
#     --is_end_to_end False --features_path ./github_ignore_material/raw_data/features.hdf5 --partial_load False \
#     --print_every_iter 11807 --eval_every_iter 999999 \
#     --reinforce False --num_epochs 30

python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533   --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 3e-5 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.3 \
    --dec_drop 0.3 --enc_input_drop 0.3 --dec_input_drop 0.3 --drop_other 0.3  \
    --batch_size 4 --num_accum 3 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path /scratch/hisan_nlp/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ./github_ignore_material/raw_data/ --partial_load True \
    --backbone_save_path ./github_ignore_material/raw_data/swin_large_patch4_window12_384_22k.pth \
    --body_save_path ./github_ignore_material/saves/checkpoint_2023-05-07-16:24:12_epoch149it186bs16_xe_.pth \
    --print_every_iter 15000 --eval_every_iter 999999 \
    --reinforce False --num_epochs 100

# python data_generator.py \
#     --save_model_path /scratch/hisan_nlp/checkpoint.pth \
#     --output_path ./github_ignore_material/raw_data/features.hdf5 \
#     --images_path ./github_ignore_material/raw_data/ \
#     --captions_path ./github_ignore_material/raw_data/

# python train.py --N_enc 3 --N_dec 3  \
#     --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
#     --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
#     --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
#     --batch_size 24 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
#     --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
#     --is_end_to_end False --partial_load True \
#     --features_path ./github_ignore_material/raw_data/features.hdf5 \
#     --body_save_path /scratch/hisan_nlp/checkpoint.pth \
#     --print_every_iter 4000 --eval_every_iter 99999 \
#     --reinforce True --num_epochs 15

# python test.py --N_enc 3 --N_dec 3 --model_dim 512 \
#     --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end False \
#     --features_path ./github_ignore_material/raw_data/features.hdf5 \
#     --save_model_path ./github_ignore_material/saves/rf.pth
