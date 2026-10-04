import argparse
from compute.config import H100

parameters = 1_000_000_000
training_tokens = 100_000_000_000
token_weight = 6
mfu = 50
gpu = "H100"


def compute_flops(parameters: int, training_tokens: int) -> int:
  global token_weight
  total_flops = token_weight * parameters * training_tokens

  return total_flops


def calculate_time(total_flops: int, gpu_flops: int, mfu: int):
  float_mfu = mfu / 100
  return total_flops / (gpu_flops * float_mfu)


def main():
  global gpu, parameters, training_tokens, mfu
  args = argparse.ArgumentParser()
  args.add_argument("--gpu", default=gpu, type=str, help="GPU")

  args = args.parse_args()
  if args.gpu.lower() == "h100":
    print("Selected GPU", args.gpu)
    h100 = H100
    gpu_flops = h100.INT8.FLOPS
    gpu_value_unit = h100.INT8.UNIT
    if gpu_value_unit.lower() == "tera":
      gpu_flops = gpu_flops * 1_000_000_000_000
      print(f"{gpu_flops:_}")
      final_flops = compute_flops(
      parameters=parameters,
      training_tokens=training_tokens
      )
      print(f"{final_flops:_}")
      total_time_in_sec = calculate_time(
      final_flops,
      gpu_flops,
      mfu
      )

      print(total_time_in_sec)



if __name__ == "__main__":
  main()
  
