import os
from ultralytics import YOLO

def exportModel(model_name: str, model_path: str, format: str, repre: str, save_dir: str):
	model = YOLO(model_path)
	if format == "ONNX":
		extension = "onnx"
	elif format == "NCNN":
		extension = "ncnn"
	elif format == "TensorRT":
		extension == "engine"

	print("---------- Exporting " + model + " to " + format + " format ----------")
	if repre == "fp16":
		model.export(format=extension, half=True)
	elif repre == "int8":
		model.export(formar=extension, int8=True)

	if not os.path.isdir(save_dir):
		print("---------- Save directory not found")
		print("---------- Creating Save directory")
		os.mkdir(save_dir + "/" + model_name + "/weights")

	os.rename(model_path.replace("pt", extension), save_dir + "/" + model_name + "/weights")
	print("---------- Model exported to " + format + " format ----------")

if __name__ == "__main__"

	parser = argparser.ArgumentParser(
			prog = 'exportModel',
			description = 'Export a Pytorch format YOLO model to other format',
			epolig = 'Prueba')
	parser.add_argument("model", type=str, help="Name of YOLO model to be exported")
	parser.add_argument("save_dir", type=str, help="Directory to save the exported model")
	parser.add_argument("--train_dir", type=str, default="models/trained/", help="Directory where Pytorch format trained models are located")
	parser.add_argument("--format", type=str, default="ONNX", help="Format of the exported model")
	parser.add_argument("--repre", type=str, help="Numeric representation of the exported model")
	args = parser.parse_args()
	exportModel(args.model, args.dir + args.model + "/weights/best.pt", args.format, args.repre, args.save_dir)

