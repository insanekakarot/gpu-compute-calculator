parameters = 1_000_000_000 #1 Billion Parameters
tokens = 10_000_000_000
weight_token = 6


def flops():
  global parameters, tokens, weight_token
  flops = parameters * tokens * weight_token

  print(f"{flops:,}")


if __name__ == "__main__":
  flops()



