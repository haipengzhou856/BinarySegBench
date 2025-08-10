accelerate launch --config_file gpu_multi.yaml --main_process_port 29050 \
main.py \
--dataset_config /vip_media/haipeng/BinarySegBench/configs/datasets/DIS5K.yaml \
--model_config /vip_media/haipeng/BinarySegBench/configs/models/Segformer.py \
--model_name Segformer \
--exp_name debug_test \
--custom_name debug \
--data_input /vip_media/haipeng/Data/DIS5K \
--data_output /vip_media/haipeng/output