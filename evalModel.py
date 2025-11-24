import time
import argparse
import threading
import numpy as np

from tqdm import tqdm
from jtop import jtop
from ultralytics import YOLO
from os import listdir, path
from contextlib import contextmanager
from typing import Optional, List, Tuple


def _getTotalPower(stats) -> Optional[float]:
	if not isinstance(stats, dict):
		return None
	val = float(stats['Power TOT'])/1000.0
	return val


class PowerSampler:

	def __init__(self, jetson, period_s: float = 0.02):
		self.jetson = jetson
		self.period_s = period_s
		self.ts: List[float] = []
		self.pw: List[float] = []
		self._run = False
		self._th: Optional[threading.Thread] = None

	def start(self):
		self.ts.clear()
		self.pw.clear()
		self._run = True
		self._th = threading.Thread(target=self._loop, daemon=True)
		self._th.start()

	def stop(self):
		self._run = False
		if self._th is not None:
			self._th.join(timeout=1.0)
			self._th = None

	def _loop(self):
		next_t = time.perf_counter()
		while self._run and self.jetson.ok():
			now = time.perf_counter()
			stats = self.jetson.stats
			p_w = _getTotalPower(stats)
			if p_w is not None:
				self.ts.append(now)
				self.pw.append(p_w)
			next_t += self.period_s
			delay = max(0.0, next_t - time.perf_counter())
			if delay > 0:
				time.sleep(delay)

	def integrate_energy_j(self) -> Tuple[float, float]:
		if len(self.ts) < 2:
			return 0.0, 0.0
		t = np.array(self.ts)
		p = np.array(self.pw)
		duration = t[-1] - t[0]
		energy = float(np.trapz(p,t))
		return energy, duration

@contextmanager
def jtop_context():
	with jtop() as jetson:
		while not jetson.ok():
			time.sleep(0.05)
		yield jetson

def run_on_images(model_name: str, model_path: str, val_path: str):
	model = YOLO(model_path)
	metrics_all = []

	with jtop_context() as jetson:
		sampler = PowerSampler(jetson, period_s = 1)
		time.sleep(0.005)
		dummy = np.zeros((640, 640, 3), dtype=np.uint8)
		x = model(dummy, verbose=False)

		predTimes = []

		print("----------   Start evaluation of " + model_name + " model evaluation   ----------")
		sampler.start()

		results = model(val_path, imgsz=640, batch=1, half=True, stream=True)
		for result in tqdm(results, total=len(listdir(val_path)), desc="Model Evaluation"):
			predTimes.append(result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess'])

		sampler.stop()
		print("----------   YOLO" + model_name + " model evaluated ----------")

		inf_time = sum(predTimes) / len(predTimes)
		energy, duration = sampler.integrate_energy_j()

		print("Average Inference Speed: ", inf_time, " ms")
		print("Average FPS: ", int(1000/inf_time))
		print("Average Energy: " , energy, " J")
		print("Average Power: ", energy/duration, " W")


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			prog = 'evalModel',
			description = 'Evaluates a YOLO Model with Ultralytics framework',
			epilog = 'Prueba')
	parser.add_argument("model", type=str, help="Name of YOLO model to be evaluated")
	parser.add_argument("--dir", type=str, default="models/exported/OrinNanoSup/", help="Directory where the model is located")
	parser.add_argument("--format", type=str, default="pt", help="Model export format")
	parser.add_argument("--data", type=str, default="datasets/DsLMF_minerBehaviorDataset/images/val", help="Directory of validation dataset")
	parser.add_argument("--power", type=str, default="None", help="Device used to measure power consumption")
	args = parser.parse_args()
	run_on_images(args.model, args.dir + args.model + "/best." + args.format, args.data)
