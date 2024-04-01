import csv
import os
from dataclasses import dataclass
from io import StringIO
from multiprocessing import Queue

from can_llamas_drive.video_worker import Video, VideoProcessWorker


@dataclass
class Params:
    """This class stores the parameters used for the video processing

    Args:
        clip_model_path: Filepath of the CLIP model
        llava_model_path: Filepath of the LLaVa model
        system_prompt: The system prompt given to the LLM
        prompt: The prompt given to the LLM
        context_length: Length of the context window
        number_of_gpu_layers: Number of layers offloaded to the GPU
        seed: The random seed
        temperature: The temperature parameter for the LLM
        number_of_frames_to_evaluate: The number of video frames used in each evaluation
        number_of_parallel_processes: The number of videos that is processed in parallel
        verbose: Print verbose output

    """

    clip_model_path: str
    llava_model_path: str
    system_prompt: str
    prompt: str
    context_length: int = 4096
    number_of_gpu_layers: int = -1
    seed: int = -1
    temperature: float = 0.1
    number_of_frames_to_evaluate: int = 3
    number_of_parallel_processes: int = 1
    verbose: bool = False


def get_video_metadata(metadata_filepath: str) -> list:
    """Get the metadata of the videos

    Args:
        metadata_filepath: Filepath to the file with metadata

    Returns:
        The video metadata

    """
    if "Crash" in metadata_filepath:
        buffer = StringIO()
        with open(metadata_filepath) as file:
            buffer.write(file.read().replace("[", "|").replace("]", "|"))

        buffer.seek(0)
        reader = csv.reader(buffer, quotechar="|")
        keys = (
            "name",
            "labels",
            "startframe",
            "youtube_id",
            "time_of_day",
            "weather",
            "ego_involved",
        )
        metadata = [dict(zip(keys, values)) for values in list(reader)]

    else:
        with open(metadata_filepath) as file:
            reader = csv.reader(file)
            ids = list(reader)

        ids = ids[0]
        metadata = [{"name": id} for id in ids]

    return metadata


def get_videos(
    input_directory: str, output_directory: str, metadata_filepath: str
) -> list[Video]:
    """Get the data of the videos that are not already processed

    Args:
        input_directory: Path to the directory containing the videos
        output_directory: Path to the directory containing the results
        metadata_filepath: Filepath of the file containing the video metadata

    Returns:
        List of Video objects to be processed

    """
    videos = []
    metadata = get_video_metadata(metadata_filepath)
    for entry in metadata:
        filepath = f"{input_directory}/{entry['name']}.mp4"
        is_crash = "labels" in entry.keys()
        is_ego_involved = entry.get("ego_involved")
        labels = entry.get("labels")
        already_processed = os.path.isfile(
            os.path.join(output_directory, "csv", entry["name"] + ".csv")
        )
        if not already_processed:
            if not is_crash or is_crash and is_ego_involved:
                videos.append(Video(filepath, is_crash, is_ego_involved, labels))

    return videos


class ProcessVideos:
    """Class that handles the (parallel) processing of videos

    Args:
        params: The parameters for video processing
    """

    def __init__(self, params) -> None:
        self.params = params

    def create_output_directory(self, output_directory: str):
        """Create output directories if not exist"""
        subdirs = ["csv", "img"]
        for subdir in subdirs:
            if not os.path.exists(os.path.join(output_directory, subdir)):
                os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

    def run(self, videos: list[Video], output_directory: str) -> list[bool]:
        """Start video processing

        Args:
            videos: List of Video objects to be processed
            output_directory: Filepath of directory where results are stored

        Returns:
            List of (trigger) results
        """
        results = []
        input_queue = Queue()
        output_queue = Queue()

        for video in videos:
            input_queue.put(video)
            
        self.create_output_directory(output_directory)

        processes = [
            VideoProcessWorker(
                input_queue, output_queue, output_directory, self.params, i
            )
            for i in range(self.params.number_of_parallel_processes)
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join()
        while not output_queue.empty():
            results.append(output_queue.get_nowait())

        return results