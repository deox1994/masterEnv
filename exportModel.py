import os
import argparse
from ultralytics import YOLO

def exportModel(model_name: str, model_path: str, format: str, repre: str, save_dir: str):
	model = YOLO(model_path)

	print("---------- Exporting " + model_name + " to " + format + " format ----------")
	if repre == "fp16":
		model.export(format=format, half=True)
	elif repre == "int8":
		model.export(format=format, int8=True)
	else:
		model.export(format=format)

	if not os.path.isdir(save_dir + "/" + model_name + "/weights"):
		print("---------- Save directory not found")
		print("---------- Creating Save directory")
		os.makedirs(save_dir + "/" + model_name + "/weights")

	os.rename(model_path.replace("pt", format), save_dir + "/" + model_name + "/weights/best." + format)
	print("---------- Model exported to " + format + " format ----------")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			prog = 'exportModel',
			description = 'Export a Pytorch format YOLO model to other format',
			epilog = 'Prueba')
	parser.add_argument("model", type=str, help="Name of YOLO model to be exported")
	parser.add_argument("save_dir", type=str, help="Directory to save the exported model")
	parser.add_argument("--train_dir", type=str, default="models/trained/", help="Directory where Pytorch format trained models are located")
	parser.add_argument("--format", type=str, default="onnx", help="Format of the exported model")
	parser.add_argument("--repre", type=str, default="fp32", help="Numeric representation of the exported model")
	args = parser.parse_args()
	exportModel(args.model, args.train_dir + args.model + "/weights/best.pt", args.format, args.repre, args.save_dir)

