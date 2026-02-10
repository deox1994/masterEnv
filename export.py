from os import path, makedirs, rename
import argparse
from ultralytics import YOLO

def exportModel(modelName: str, modelPath: str, format: str, half: bool, saveDir: str):
	model = YOLO(modelPath)

	print("---------- Exporting " + modelName + " to " + format + " format ----------")
	model.export(format=format, half=half)

	if not path.isdir(path.join(saveDir, modelName, "weights")):
		print("---------- Save directory not found")
		print("---------- Creating Save directory")
		makedirs(path.join(saveDir, modelName, "weights"))

	rename(modelPath.replace("pt", format), path.join(saveDir, modelName, "weights", "best." + format))
	print("---------- Model exported to " + format + " format ----------")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			prog = 'exportModel',
			description = 'Export a Pytorch format YOLO model to other format with Ultralytics framework',
			epilog = 'Export complete')
	parser.add_argument("modelName", type=str, help="Name of YOLO model to be exported")
	parser.add_argument("saveDir", type=str, help="Directory to save the exported model")
	parser.add_argument("--modelDir", type=str, default=path.join("models", "trained"), help="Directory where Pytorch format trained models are located")
	parser.add_argument("--format", type=str, default="onnx", help="Format of the exported model")
	parser.add_argument("--half", action="store_true", default=False, help="Numeric representation of the exported model")
	args = parser.parse_args()
	exportModel(args.modelName, path.join(args.modelDir, args.modelName, "weights", "best.pt"), args.format, args.half, args.saveDir)

