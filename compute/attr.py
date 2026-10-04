import json
import argparse
from typing import List, Dict
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
json_file = project_root / "data" / "gpus.json"
gpu = "nvidia h100"
quantization_unit = "float64"
gpu_count = 1


def load_json(path: Path):
  with open(path) as file:
    return json.load(file)


def value_unit_conversion(value: int):
  if value is None or value == "":
    return ValueError("Value not entered")

  if value < 1_000:
    return float(value), "Ones"

  units = ["Thousands", "Million", "Billion", "Trillion", "Quadrillion", "Qunitillion", "Sextillion", "Septillion", "Octillion", "Nonillion", "Decillion"]
  for unit in units:
    if value >= 1_000:
      value = value / 1_000
      final_unit = unit

    else:
      break

  return round(value, 2), final_unit



def extract_gpu_specs(
data: List[Dict[str, any]],
q_key: str,
gpu: str,
flops_unit_required: bool = True
):
  for item in data:
    if gpu.lower() == item:
      flops = data[gpu.lower()][q_key]["flops"]
      flops_unit = data[gpu.lower()][q_key]["flops_unit"]

      if flops_unit_required:
        return flops, flops_unit

      return flops


def gpu_unit_conversion(flops_unit: str, gpu_flops: int):
  if flops_unit.lower() == "tera":
    a_gpu_flops = gpu_flops * 1_000_000_000_000
    return a_gpu_flops

  elif flops_unit.lower() == "giga":
    a_gpu_flops = gpu_flops * 1_000_000_000
    return a_gpu_flops

  elif flops_unit.lower() == "mega":
    a_gpu_flops = gpu_flops * 1_000_000
    return a_gpu_flops
    

def compute_flops(parameters: int, training_tokens: int, token_weight: int = 6) -> int:
  return token_weight * parameters * training_tokens
  

def time_taken(gpu_flops: int, computed_flops: int, mfu: int = 50, gpu_count: int = 1):
  mfu = mfu / 100
  if gpu_flops <= 0:
    return 0
  return round(computed_flops / (gpu_count * mfu * gpu_flops), 2)


class TimeConversion():
  # Seconds Unit
  def secs_into_minutes(seconds: float):
    return round(seconds / 60, 2)

  def secs_into_hours(seconds: float):
    return round(seconds / 3600, 2)

  def secs_into_days(seconds: float):
    return round((seconds / 3600) / 24, 2)

  def secs_into_weeks(seconds: float):
    return round(((seconds / 3600) / 24) / 7, 2)


def unit_multiplier(unit: str, value: float):
  units = {"M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000, "QD": 1_000_000_000_000_000}
  for x, y in units.items():
    if x.lower() == unit.lower():
      result = y * value
      return result


def main():
  global gpu, quantization_unit
  args = argparse.ArgumentParser()
  args.add_argument("--gpu", default=gpu, type=str)
  args.add_argument("--gpu-count", default=gpu_count, type=int)
  args.add_argument("--quantization", default=quantization_unit, type=str)
  args.add_argument("--parameters", default=1_000_000_000, type=int)
  args.add_argument("--tokens", default=20_000_000_000, type=int)
  args.add_argument("--mfu", default=60, type=int)

  args = args.parse_args()
  data = load_json(json_file)

  tc = TimeConversion

  gpu_flops, gpu_flops_unit = extract_gpu_specs(data, args.quantization, args.gpu, flops_unit_required=True)
  computed_flops = compute_flops(args.parameters, args.tokens)
  actual_gpu_flops = gpu_unit_conversion(gpu_flops_unit, gpu_flops)
  computed_flops_in_eng, computed_flops_unit_in_eng = value_unit_conversion(computed_flops)
  if actual_gpu_flops == 0:
    print(f"This {args.gpu} GPU isn't supporting {args.quantization} Quantization")
    return
  gpu_flops_in_eng, gpu_flops_unit_in_eng = value_unit_conversion(actual_gpu_flops)
  training_time_taken = time_taken(actual_gpu_flops, computed_flops, args.mfu, args.gpu_count)
  training_time_in_minutes = tc.secs_into_minutes(training_time_taken)
  training_time_in_hours = tc.secs_into_hours(training_time_taken)
  training_time_in_days = tc.secs_into_days(training_time_taken)
  training_time_in_weeks = tc.secs_into_weeks(training_time_taken)

  print("" * 50)
  print(f"Selected GPU                                            {args.gpu.upper()}")
  print(f"Selected Quantization                                   {args.quantization}")
  print(f"Total Computed Flops are in math:                       {computed_flops}")
  print(f"Total Computed Flops are in English:                    {computed_flops_in_eng} {computed_flops_unit_in_eng}")
  print(f"Actual GPU Flops are in math:                           {actual_gpu_flops}")
  print(f"GPU Flops in English                                    {gpu_flops_in_eng} {gpu_flops_unit_in_eng}")
  print(f"How much time it will take in seconds                   {training_time_taken} Seconds")
  print(f"How much time it will take in minutes                   {training_time_in_minutes} Minutes")
  print(f"How much time it will take in hours                     {training_time_in_hours} Hours")
  print(f"How much time it will take in days                      {training_time_in_days} Days")
  print(f"How much time it will take in weeks                     {training_time_in_weeks} Weeks")


def load_precisions(file: Path):
  data = load_json(file)

  for gpu, precisions in data.items():
    gpu = gpu
    precisions = precisions

  return list(precisions.keys())


def load_gpus(file: Path):
  data = load_json(file)

  return list(data.keys())
  
  

if __name__ == "__main__":
  main()
