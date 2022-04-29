from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

def model():
    # Setup a model config (recipe for training a Detectron2 model)
    cfg=get_cfg()

    # Add some basic instructions for the Detectron2 model from the model_zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

    # Add some pretrained model weights from an object detection model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    # Setup datasets to train/validate on (this will only work if the datasets are registered with DatasetCatalog)
    cfg.DATASETS.TRAIN = ("cmaker-fireplace-train",)
    cfg.DATASETS.TEST = ("cmaker-fireplace-valid",)

    # How many dataloaders to use? This is the number of CPUs to load the data into Detectron2, Colab has 2, so we'll use 2
    cfg.DATALOADER.NUM_WORKERS = 2

    # How many images per batch? The original models were trained on 8 GPUs with 16 images per batch, since we have 1 GPU: 16/8 = 2.
    cfg.SOLVER.IMS_PER_BATCH = 2

    # We do the same calculation with the learning rate as the GPUs, the original model used 0.01, so we'll divide by 8: 0.01/8 = 0.00125.
    cfg.SOLVER.BASE_LR = 0.00125

    # How many iterations are we going for? (300 is okay for our small model, increase for larger datasets)
    cfg.SOLVER.MAX_ITER = 300

    # ROI = region of interest, as in, how many parts of an image are interesting, how many of these are we going to find? 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # We're only dealing with 2 classes (coffeemaker and fireplace) 
    cfg.MODEL.RETINANET.NUM_CLASSES = 2

    # Setup output directory, all the model artefacts will get stored here in a folder called "outputs" 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Setup the default Detectron2 trainer, see: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
    trainer = DefaultTrainer(cfg)

    # Resume training from model checkpoint or not, we're going to just load the model in the config: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer.resume_or_load
    trainer.resume_or_load(resume=False) 

    # Start training
    trainer.train()

    return trainer

def predictions(cfg):
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 

    
    cfg.DATASETS.TEST = ("cmaker-fireplace-valid",) 

    predictor = DefaultPredictor(cfg)

    return predictor

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 

if __name__ == "__main__":
    model = model()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
    predictions(cfg)