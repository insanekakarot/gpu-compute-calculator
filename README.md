

# Model Training Calculator App

A Python web application for calculating and comparing GPU metrics.

## Architecture
* **Frontend:** HTML templates with modular components (`/templates`)
* **Backend:** Python web framework (`app.py`)
* **Compute:** Custom calculation logic (`calculator.py` & `/compute`)
* **Data:** Local JSON storage (`/data/gpus.json`)

## Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the testing suite: `python test.py`
4. Start the development server: `python app.py`

After running!
5. /* Go to http://0.0.0.0:8000 to use tool.



### Using CLI
You can use CLI version directly in your command prompt, Without installing packages. Just pure python logics.
** Python File CLI Use**
`python compute/attr.py`

** Arguments **
1. `--parameters` 1_000_000 - Parameters of Model! ** Type = Integer **
2. `--tokens` 200_000_000 - Training Tokens, How much tokens are in your model! ** Type = Integer **
3. `--quantization` "float16" - Precision of model training! ** Type = String **
4. `--mfu` 40 - MFU, GPU Processing Percentage! ** Type = Integer **
5. `--gpu-count` 4 - How many GPUs could train your model! ** Type = Integer **
6. `--gpu` "nvidia h100" - Which GPU you want to use? ** Type = String **



