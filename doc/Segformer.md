# Segformer (NIPS21)

## BIGSHA

Modify the BIGSHA dataset root in the `config/Segformer.yml`

**Training Command:**

```sh
accelerate launch --config_file accelerate_cfg.yaml main.py --config "./config/Segformer.yml" --manual_flag "BIGSHA_EXP"
```

**Inference Command**

Download and unzip the ckpt file ([Google Drive(TODO)]()), change the path in the `config/Segformer.yml`: `MODEL--CKP_PATH: xxx`,  then run

```sh
accelerate launch --config_file accelerate_cfg.yaml inference.py --config "./config/Segformer.yml" --manual_flag "BIGSHA_Infer"
```

**Check with me**

My implement details about Logs is in ([Google Drive(TODO)]())



## Trans10K

TODO

