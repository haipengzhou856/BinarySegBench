accelerate launch \
  --config_file gpu_multi.yaml \    # Configuration file for multi-GPU training
  --main_process_port 29050 \       # Port for the main process
  main.py \
  --dataset_config ./configs/datasets/DIS5K.yaml \
  --model_config ./configs/models/Segformer.py \
  --model_name Segformer \
  --exp_name debug_test \            # Experiment name
  --custom_name
  --data_input /home/haipeng/Data/DIS5K \
  --data_output /home/haipeng/output
