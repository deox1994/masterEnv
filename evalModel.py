import sys
import time
import serial
import argparse
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
from os import listdir, path
from typing import Optional, List, Tuple


def run_on_images(model_name: str, model_path: str, data_path: str, repetitions:int, half:bool, power: bool, serPort: str):
	model = YOLO(model_path)

	print("----------   Start evaluation of " + model_name + " model evaluation   ----------")

	print("----------   Step 1: Evaluation Performance of the model   ----------")

	metrics = model.val(data="../datasets/DsLMF_minerBehaviorDataset" + ".yaml", imgsz=640, batch=1, verbose=False, half=half)

	print("----------   Step 2: Evaluating Inference time and Power Consumption of the model   ----------")


	if power:
		try:
			ser = serial.Serial(port=serPort, baudrate=115200)
			ser.isOpen()
			mess = "S"
			ser.write(mess.encode())
		except:
			print("Can not open Serial Port")

	infTimes = []
	for i in range(repetitions):
		predTimes = []
		results = model(data_path + "/images/val", imgsz=640, batch=1, stream=True, verbose=False, half=half)
		for result in tqdm(results, total=len(listdir(data_path + "/images/val")), desc="Model Evaluation - Experiment " + str(i+1)):
			predTimes.append(result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess'])
		infTimes.append(sum(predTimes) / len(predTimes))

	print("----------   YOLO" + model_name + " model evaluated ----------")

	print("Mean Average Precision mAP@50: ", metrics.box.map50)
	print("Mean Average Precision mAP@50:95: ", metrics.box.map)
	avg_infTime = sum(infTimes) / len(infTimes)
	print("Average Inference Speed: ", avg_infTime, " ms")
	print("Average FPS: ", int(1000/avg_infTime))

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
	parser.add_argument("--data", type=str, default="datasets/DsLMF_minerBehaviorDataset", help="Directory of dataset")
	parser.add_argument("--format", type=str, default="pt", help="Model export format")
	parser.add_argument("--repet", type=int, default="5", help="Repetitions to average inference time and power consumption")
	parser.add_argument("--half", action="store_true", help="Activates 16 bit (half) Floting point representation for all evaluations")
	parser.add_argument("--power", action="store_true", help="Activates energy and power measurements with Arduino controller")
	parser.add_argument("--serPort", type=str, default="None", help="Device port where Arduino is connected to measure power consumption")
	args = parser.parse_args()
	run_on_images(args.model, args.dir + args.model + "/weights/best." + args.format, args.data, args.repet, args.half, args.power, args.serPort)
