################ steal this setting from Dassl and Detectron2 ################
######################### big thanks to the authors ##########################

from yacs.config import CfgNode as CN

#####################  Env config #####################
_C = CN()
_C.SEED = 42
_C.OUTPUT_DIR = '/home/haipeng/output'
_C.LOG = '/log_path'  # logs
_C.TB = '/tb_path'  # tensorboard
_C.CKPT = '/ckpt_path'  # checkpoints
_C.RESULT = '/result_path'  # result
_C.EXP_NAME = 'debug'
_C.RESUME_DIR = ''

#####################  Input config #####################
_C.INPUT = CN()
_C.INPUT.SIZE = 518
_C.INPUT.INTERPOLATION = "bilinear"
#_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406] # I've fixed in dataset
#_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]  # I've fixed in dataset

#####################  Dataset config #####################
_C.DATASET = CN()
_C.DATASET.ROOT = ""
_C.DATASET.NAME = ""
_C.DATASET.NUM_CLASSES = 1

#####################  Dataloader config #####################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 12
#####################  Model config #####################
## define all the model here
_C.MODEL = CN()
_C.MODEL.Segformer = CN()
_C.MODEL.SAM2 = CN()
_C.MODEL.ICCV = CN()
_C.MODEL.DPT = CN()

_C.MODEL.NAME = 'Segformer'

#####################  Solver config #####################
_C.SOLVER = CN()
_C.SOLVER.LR = 0.000003
_C.SOLVER.EPOCH = 30
_C.SOLVER.STORE_EPOCH = 2
_C.SOLVER.OPTIMIZER = 'AdamW'
_C.SOLVER.SGD_MOMENTUM = 0.9
_C.SOLVER.DECAY = 0.001

_C.SOLVER.SCHEDULER = 'Linear'
_C.SOLVER.STEP_GAMMA = 0.8
