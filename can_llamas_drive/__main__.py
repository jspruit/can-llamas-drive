import argparse
import os

from can_llamas_drive.process_videos import Params, ProcessVideos, get_videos

DEFAULT_PARAMS = Params(
    clip_model_path="./models/llava-v1.5-7b/mmproj-model-f16.gguf",
    llava_model_path="./models/llava-v1.5-7b/ggml-model-Q5_K_M.gguf",
    system_prompt="You are the assitant of the driver of a car. When the driver fails to brake in a dangerous situation, you can perform an emergency stop. You are only expected to perform an emergency stop if an immediate collision is unavoidable, unnecessary emergency stops should be avoided as much as possible. The data you get are frames from the dashcam feed of the vehicle of the past 0.5 s from old to new, the interval between the frames is 0.1 s. Use the relative difference between the frames to estimate the velocities and behaviour for all road users relative to our car.",
    prompt="Based on the current frame and previous ones, do you trigger an emergency stop or not? Answer 'Yes' or 'No'.",
    context_length=4096,  # Prompt token size = 3046
    number_of_gpu_layers=-1,
    seed=-1,
    temperature=0.1,
    number_of_frames_to_evaluate=1,
    number_of_parallel_processes=1,
    verbose=False,
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="can_llamas_drive",
        description="Evaluate LLM performance on dashcam videos",
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to the CarCrashDataset, e.g. ./datasets/CarCrash",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path the directory where results will be stored",
    )
    parser.add_argument(
        "-cm",
        "--clip-model",
        dest="clip_model_path",
        type=str,
        default=DEFAULT_PARAMS.clip_model_path,
        help="Path to the clip model",
    )
    parser.add_argument(
        "-lm",
        "--llava-model",
        dest="llava_model_path",
        type=str,
        default=DEFAULT_PARAMS.llava_model_path,
        help="Path to the Llama model",
    )
    parser.add_argument(
        "-sp",
        "--system-prompt",
        type=str,
        default=DEFAULT_PARAMS.system_prompt,
        help="The system prompt (i.e. what the LLM is told at the start of conversation)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=DEFAULT_PARAMS.prompt,
        help="The prompt (i.e. what the LLM is told)",
    )
    parser.add_argument(
        "-c",
        "--context-length",
        type=int,
        default=DEFAULT_PARAMS.context_length,
        help="Length of the context that is evaluated by the LLM",
    )
    parser.add_argument(
        "-ngl",
        "--number-of-gpu-layers",
        type=int,
        default=DEFAULT_PARAMS.number_of_gpu_layers,
        help="Number of layers that are offloaded to the GPU",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_PARAMS.seed,
        help="RNG seed (use -1 for random seed)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=DEFAULT_PARAMS.temperature,
        help="Temperature (advised value = 0.1)",
    )
    parser.add_argument(
        "-nf",
        "--number-of-frames",
        dest="number_of_frames_to_evaluate",
        type=int,
        metavar="N",
        default=DEFAULT_PARAMS.number_of_frames_to_evaluate,
        help="Number of video frames that will be evaluated on each cycle",
    )
    parser.add_argument(
        "-par",
        "--parallel",
        dest="number_of_parallel_processes",
        type=int,
        metavar="N",
        default=DEFAULT_PARAMS.number_of_parallel_processes,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DEFAULT_PARAMS.verbose,
        help="Print debug info",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    process_videos = ProcessVideos(args)

    metadata_filepath = os.path.join(args.input_directory, "Crash-1500.txt")
    input_directory = os.path.join(args.input_directory, "Crash-1500")
    output_directory = os.path.join(args.output_directory, "crash")
    crash_videos = get_videos(input_directory, output_directory, metadata_filepath)
    results_crash = process_videos.run(crash_videos, output_directory)
    print(results_crash)

    metadata_filepath = os.path.join(args.input_directory, "Normal-1500.txt")
    input_directory = os.path.join(args.input_directory, "Normal-1500")
    output_directory = os.path.join(args.output_directory, "normal")
    normal_videos = get_videos(input_directory, output_directory, metadata_filepath)
    results_crash = process_videos.run(normal_videos, output_directory)
    print(results_crash)


main()
