from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=r"G:\RYD\DataSet\new")
trainer.setTrainConfig(object_names_array=["question"], batch_size=4, num_experiments=100, train_from_pretrained_model=r"G:\ryd\yolo\pretrained-yolov3.h5")
trainer.trainModel()
