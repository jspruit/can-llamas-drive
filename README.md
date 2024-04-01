# Can llamas drive?

This repo evaluates how well the LLaVA 1.5 multimodal large language model would perform when used as an Autonomous Emergency Braking System (AEBS) in a vehicle.

_Can llamas drive?_ evaluates videos from the [CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset) in a frame-by-frame fashion using the LLaVA multimodal large language model. For each frame (or sequence of frames), the LLM is prompted and asked if an emergency stop is required based on the information in the video frame. The [CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset) consist of two sets of videos. One set contains normal videos (negatives), while the other set contains videos where crashes happen (positives). 

## How to use
Of course it's probably best to first create a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install required packages to run the `playground.ipynb` and `results.ipynb` notebooks:
```
pip install -r requirements.txt
```

Install `can_llamas_drive`:
```
pip install -e .
```

Download the LLaVA 1.5 7b model to the `./models/` directory:
```
wget https://huggingface.co/granddad/llava-v1.5-7b-gguf/resolve/main/ggml-model-Q5_K_M.gguf -P ./models/llava-v1.5-7b/

wget https://huggingface.co/granddad/llava-v1.5-7b-gguf/resolve/main/mmproj-model-f16.gguf -P ./models/llava-v1.5-7b/
```

Or the LLaVA 1.5 13b model:
```
wget https://huggingface.co/granddad/llava-v1.5-13b-gguf/resolve/main/ggml-model-Q5_K_M.gguf -P ./models/llava-v1.5-13b/

wget https://huggingface.co/granddad/llava-v1.5-13b-gguf/resolve/main/mmproj-model-f16.gguf -P ./models/llava-v1.5-13b/
```

Download and extract the CarCrashDataset from https://github.com/Cogito2012/CarCrashDataset and put the downloaded folder in `./datasets`, i.e. you should end up with `./datasets/CarCrash`.

Use `playground.ipynb` to test how the prompt given to LLaVA influences the output for a single video frame.

Whenever you want test the prompt on the entire dataset run:
```
usage: can_llamas_drive [-h] [-cm CLIP_MODEL_PATH] [-lm LLAVA_MODEL_PATH] [-sp SYSTEM_PROMPT] [-p PROMPT] [-c CONTEXT_LENGTH] [-ngl NUMBER_OF_GPU_LAYERS] [-s SEED] [-t TEMPERATURE]
                        [-nf N] [-par N] [--verbose]
                        input_directory output_directory

Evaluate LLM performance on dashcam videos

positional arguments:
  input_directory       Path to the CarCrashDataset, e.g. ./datasets/CarCrash
  output_directory      Path the directory where results will be stored

options:
  -h, --help            show this help message and exit
  -cm CLIP_MODEL_PATH, --clip-model CLIP_MODEL_PATH
                        Path to the clip model
  -lm LLAVA_MODEL_PATH, --llava-model LLAVA_MODEL_PATH
                        Path to the Llama model
  -sp SYSTEM_PROMPT, --system-prompt SYSTEM_PROMPT
                        The system prompt (i.e. what the LLM is told at the start of conversation)
  -p PROMPT, --prompt PROMPT
                        The prompt (i.e. what the LLM is told)
  -c CONTEXT_LENGTH, --context-length CONTEXT_LENGTH
                        Length of the context that is evaluated by the LLM
  -ngl NUMBER_OF_GPU_LAYERS, --number-of-gpu-layers NUMBER_OF_GPU_LAYERS
                        Number of layers that are offloaded to the GPU
  -s SEED, --seed SEED  RNG seed (use -1 for random seed)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature (advised value = 0.1)
  -nf N, --number-of-frames N
                        Number of video frames that will be evaluated on each cycle
  -par N, --parallel N  Number of parallel processes
  --verbose             Print debug info
```

For example:
```
python -m can_llamas_drive ./datasets/CarCrash/videos ./output
--clip-model ./models/llava-v1.5-7b/mmproj-model-f16.gguf \
--llava-model ./models/llava-v1.5-7b/ggml-model-Q5_K_M.gguf \
--system-prompt "You are the assitant of the driver of a car. When the driver fails to brake in a dangerous situation, you can perform an emergency stop." \
--prompt "Based on the current frame and previous ones, do you trigger an emergency stop or not? Answer 'Yes' or 'No'." \
--seed 1337 \
--number-of-gpu-layers -1 \
--number-of-frames 1
```

## Performance evaluation

For now, the notebook `results.ipynb` can be used to process the results.

The CarCrashDataset consist of two sets of videos. One set contains normal videos (negatives), while the other set contains crashes (positives). Based on if and when the LLM decides to trigger an emergency stop, the video is assigned one of the following labels:
- **True positive:** The video is a crash video and there is a stop triggered by the LLM when the time-to-collision is: $0 < t_\text{TTC} < 3 s$
- **False negative:** The video is a crash video and there is no stop triggered by the LLM
- **False positive:** The video is a crash video and there is a stop triggered when $t_\text{TTC} > 3 s$ or the video is a normal video and there is a stop triggered
- **True negative:** The video is a normal video and there is no stop triggered

With this data we can create a standard confusion matrix:

<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan=2><b>Predicted condition<br>(What the LLM says)</b></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td><i>Positive</i></td>
    <td><i>Negative</i></td>
  </tr>
  <tr>
    <td rowspan=2><b>Actual condition<br>(What happens in the video)</b></td>
    <td><i>Positive</i></td>
    <td>True positive<br>(TP)</td>
    <td>False negative<br>(FN)</td>
  </tr>
  <tr>
    <td><i>Negative</i></td>
    <td>False positive<br>(FP)</td>
    <td>True negative<br>(TN)</td>
  </tr>
</table>

If the LLM's performance would be perfect, all crash videos should be true positives and all normal videos should be true negatives.

>TODO(JS): Expand
