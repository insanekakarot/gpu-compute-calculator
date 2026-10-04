from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from compute.attr import compute_flops, value_unit_conversion, extract_gpu_specs, load_json, time_taken, gpu_unit_conversion, TimeConversion, load_precisions, load_gpus, unit_multiplier
import uvicorn


json_file = "data/gpus.json"

app = FastAPI(
  doc_url=None,
  redoc_url=None,
  openapi_url=None
  )

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

tc = TimeConversion

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
  precisions = load_precisions(json_file)
  gpus = load_gpus(json_file)
  
  return templates.TemplateResponse(
  request=request, 
  name="index.html",
  context={"gpus": gpus, "precisions": precisions}
  )


@app.post("/")
def form(
parameters: int = Form(name="parameters"),
parameters_unit: str = Form(name="parameters_unit"),
tokens: int = Form(name="tokens"), 
tokens_unit: str = Form(name="tokens_unit"),
gpu: str = Form(name="gpu"), 
quantization: str = Form(name="quantization"),
mfu: int = Form(name="mfu"),
gpu_count: int = Form(name="gpu_count")
):

  actual_parameters = unit_multiplier(parameters_unit, parameters)
  actual_tokens = unit_multiplier(tokens_unit, tokens)

  computed_flops = compute_flops(actual_parameters, actual_tokens)
  p_value, p_unit = value_unit_conversion(actual_parameters)
  t_value, t_unit = value_unit_conversion(actual_tokens)

  loaded_data = load_json(json_file)


  gpu_flops, gpu_flops_unit = extract_gpu_specs(loaded_data, quantization, gpu, flops_unit_required=True)
  if gpu_flops <= 0:
    return {
    "not supported": f"{quantization} Quantization not supported with this {gpu} GPU",
    "Selected GPU": f"{gpu.upper()}",
    "Selected Precision": f"{quantization}"
    }
    
  actual_gpu_flops = gpu_unit_conversion(gpu_flops_unit, gpu_flops)
  training_time_taken = time_taken(actual_gpu_flops, computed_flops, mfu, gpu_count)
  training_time_in_minutes = tc.secs_into_minutes(training_time_taken)
  training_time_in_hours = tc.secs_into_hours(training_time_taken)
  training_time_in_days = tc.secs_into_days(training_time_taken)
  training_time_in_weeks = tc.secs_into_weeks(training_time_taken)
  actual_gpu_flops_in_count, actual_gpu_flops_unit_in_eng = value_unit_conversion(actual_gpu_flops)
  
  
  result = [{
    "Selected GPU": f"{gpu}",
    "Parameters": f"{p_value} {p_unit}",
    "Training Tokens": f"{t_value} {t_unit}",
    "Precision": f"{quantization}",
    "Training Time taken": f"{training_time_taken} Seconds",
    "Training time taken in minutes": f"{training_time_in_minutes} Minutes",
    "Training time taken in hours": f"{training_time_in_hours} Hours",
    "Training Time taken in days": f"{training_time_in_days} Days",
    "Training time taken in weeks": f"{training_time_in_weeks} Weeks",
    "GPU Total Flops": f"{actual_gpu_flops} {actual_gpu_flops_in_count} {actual_gpu_flops_unit_in_eng}",
    "GPU Flops Unit": f"{gpu_flops_unit}"
  }]


  return result

   

if __name__ == "__main__":
  uvicorn.run("app:app", host="0.0.0.0", port=8000)
