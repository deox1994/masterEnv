import sys
import time
import serial
import argparse
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
from os import listdir, path
from typing import Optional, List, Tuple


def run_on_images(model_name: str, model_path: str, val_path: str, power: str, serPort: str):
	model = YOLO(model_path)

	print("----------   Start evaluation of " + model_name + " model evaluation   ----------")

	predTimes = []

	if power:
		try:
			ser = serial.Serial(port=serPort, baudrate=115200)
			ser.isOpen()
			mess = "S"
			ser.write(mess.encode())
		except:
			print("Can not open Serial Port")

	results = model(val_path, imgsz=640, batch=1, half=True, stream=True, verbose=False)
	for result in tqdm(results, total=len(listdir(val_path)), desc="Model Evaluation"):
		predTimes.append(result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess'])

	print("----------   YOLO" + model_name + " model evaluated ----------")

	inf_time = sum(predTimes) / len(predTimes)
	print("Average Inference Speed: ", inf_time, " ms")
	print("Average FPS: ", int(1000/inf_time))

	if power:
		try:
			mess = "F"
			ser.write(mess.encode())
			time.sleep(0.1)
			if ser.inWaiting() > 0:
				power = ser.readline().decode().strip().split(',')
				print("Total Energy: ", power[0], " mJ")
				print("Average Power: ", power[1], " mW")
		except:
			print("Can not send message")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			prog = 'evalModel',
			description = 'Evaluates a YOLO Model with Ultralytics framework',
			epilog = 'Prueba')
	parser.add_argument("model", type=str, help="Name of YOLO model to be evaluated")
	parser.add_argument("dir", type=str, help="Directory where the model is located")
	parser.add_argument("--format", type=str, default="pt", help="Model export format")
	parser.add_argument("--data", type=str, default="datasets/DsLMF_minerBehaviorDataset/images/val", help="Directory of validation dataset")
	parser.add_argument("--power", action="store_true", help="Activates energy and power measurements with Arduino controller")
	parser.add_argument("--serPort", type=str, default="None", help="Device port where Arduino is connected to measure power consumption")
	args = parser.parse_args()
	run_on_images(args.model, args.dir + args.model + "/weights/best." + args.format, args.data, args.power, args.serPort)
