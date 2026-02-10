import os
import argparse
from ultralytics import YOLO

def trainModel(model: str, scale: str, name: str, yamlpath: str, transf: bool):
	
    archsPath = os.path.join('models', 'architectures')
    weightsPath = os.path.join('models', 'weights')
    if model == '5':
        architecture = os.path.join(archsPath, 'yolov' + str(model) + scale + '.yaml')
        weights = os.path.join(weightsPath, 'yolov' + str(model) + scale + 'u.pt') if transf else None
    else:
        architecture = os.path.join(archsPath, 'yolov' + str(model) + scale + '.yaml')
        weights = os.path.join(weightsPath, 'yolov' + str(model) + scale + '.pt') if transf else None
    
    model = YOLO(architecture).load(weights) if transf else YOLO(architecture)

    model.train(workers=1, device=0, data=yamlpath, epochs=200, lr0=0.01, lrf=0.01, momentum=0.937, batch=16, optimizer='SGD', plots =True, deterministic = False, project=os.path.join('models', 'trained'), name=name, close_mosaic=0, save_period=10, patience=20, verbose=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog = 'trainModel',
        description = 'Trains a YOLO model with given parameters with Ultralytics framework',
        epilog = 'Training complete')
    parser.add_argument("model", type=str, help="Version of YOLO model to be trained: 5, 8, 9, 10, 11")
    parser.add_argument("scale", type=str, help="Scale of the YOLO model architecture: n, s, m, l, x or t for YOLOv9")
    parser.add_argument("name", type=str, help="Name of the trained model")
    parser.add_argument("yamlpath", type=str, help="Path to the dataset yaml file")
    parser.add_argument("--transf", action="store_true", help="Enables training with transfer learning")
    args = parser.parse_args()
    trainModel(args.model, args.scale, args.name, args.yamlpath, args.transf)