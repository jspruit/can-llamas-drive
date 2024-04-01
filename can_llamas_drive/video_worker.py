import array
import csv
import ctypes
import os
import time
from multiprocessing import Process

import cv2
import numpy as np
from llama_cpp import Llama, llava_cpp
from llama_cpp._utils import suppress_stdout_stderr

from can_llamas_drive.my_chat_format import MyLlava15ChatHandler


class Video:
    """Class describing a Video object

    Args:
        filepath: Filepath of the video file
        is_crash: Does a crash happen in the video
        is_ego_involved: Is the ego-vehicle involved in the crash
        labels: Labels describing if a crash is happening in the corresponding video frame

    """

    def __init__(
        self,
        filepath: str,
        is_crash: bool,
        is_ego_involved: bool,
        labels: list[bool] | None = None,
    ) -> None:
        self.filepath = filepath
        self.name = (
            os.path.splitext(os.path.basename(filepath))[0]
            if os.path.isfile(filepath)
            else None
        )
        self.labels = labels
        self.is_crash = is_crash
        self.is_ego_involved = is_ego_involved

    def get_frames(self, border: bool = False) -> list:
        """Return all frames from video

        Load, convert to Numpy array, and return all video frames

        Args:
            border: Add white border of _border_ px around the video frame
        Returns:
            List with each video frame as Numpy array
        """
        frames = []
        cap = cv2.VideoCapture(self.filepath)
        cap.set(cv2.CAP_PROP_FPS, 10)
        ret, img = cap.read()
        while ret:
            if border:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                border_size = 10
                img = cv2.copyMakeBorder(
                    img,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
            frames.append(img)
            ret, img = cap.read()

        return frames


class VideoProcessWorker(Process):
    """Multiprocessing wrapper for VideoWorker class

    Args:
        input_queue: The input queue
        output_queue: The output queue
        output_directory: Filepath of the directory containing the results
        params: The parameters for the video processing
        i: The number of the VideoWorker instance
    """

    def __init__(
        self, input_queue, output_queue, output_directory, params, i
    ):
        Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.output_directory = output_directory
        self.params = params
        self.i = i

    def run(self):
        """Run the video processing"""
        while True:
            if self.input_queue.empty():
                break
            else:
                video = self.input_queue.get()
                result = VideoWorker(
                    video, self.params, self.i
                ).run_recording(self.output_directory)
                self.output_queue.put(result)


class VideoWorker:
    """Evaluate videos with LLM
    Args:
        video: The video to be evaluated
        i: The number of the VideoWorker instance

    """

    def __init__(self, video: Video, params, i: int = 0):
        self.video = video
        self.params = params
        self.i = i

        self._llava_cpp = llava_cpp
        self._frames = self.video.get_frames()
        self._embedded_frames = {}
        # self.embedded_frames = None
        self._chat_handler = MyLlava15ChatHandler(
            self._llava_cpp, self.params.clip_model_path, verbose=self.params.verbose
        )
        self._llm = Llama(
            model_path=self.params.llava_model_path,
            chat_handler=self._chat_handler,
            n_ctx=self.params.context_length,
            n_gpu_layers=self.params.number_of_gpu_layers,
            seed=self.params.seed,
            logits_all=True,
            verbose=self.params.verbose,
        )

    def _print_progress(self, current_frame, answer, duration):
        """Print the processing status"""
        print(
            f"{self.video.name} - ({current_frame - self.params.number_of_frames_to_evaluate + 1}/{len(self._frames) - self.params.number_of_frames_to_evaluate}):{answer['choices'][0]['message']['content']} - took {duration:.3f} s (Worker {self.i})"
        )

    def run_recording(self, output_directory: str) -> list[bool]:
        """Run the entire recording and save results

        Args:
            output_directory: Filepath to the directory containing the results

        Returns:
            List of booleans corresponding to the video frames. True if a stop was
            triggered by the LLM, False otherwise.

        """
        n_frames = self.params.number_of_frames_to_evaluate
        initial_trigger = True
        triggers = []
        for i in range(n_frames, len(self._frames)):
            start_time = time.time()
            answer = self.get_answer(i)
            self._print_progress(i, answer, time.time() - start_time)
            trigger = (
                1
                if answer["choices"][0]["message"]["content"].lower().strip() == "yes"
                else 0
            )
            triggers.append(trigger)

            if trigger and initial_trigger:
                cv2.imwrite(
                    f"{output_directory}/img/{self.video.name}.jpg",
                    self.get_concatenated_frames(i),
                )
                initial_trigger = False

            self._llava_cpp.llava_image_embed_free(
                self._embedded_frames[min(self._embedded_frames.keys())]
            )
            self._embedded_frames.pop(min(self._embedded_frames.keys()))
            # self._llava_cpp.llava_image_embed_free(self.embedded_frames)

        # Add initial (unprocessed) frames
        triggers = [0] * n_frames + triggers
        with open(f"{output_directory}/csv/{self.video.name}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(triggers)

        for embedded_frame in self._embedded_frames.values():
            self._llava_cpp.llava_image_embed_free(embedded_frame)
        self._embedded_frames.clear()
        #self._llava_cpp.clip_free(self._chat_handler.clip_ctx)
        self._llm.reset()

        return triggers

    def get_answer(self, current_frame_index: int):
        """Evaluate current frame

        Send system prompt, embedded frames, and prompt to LLM and wait for
        answer

        Args:
            current_frame_index: The index of the current frame

        Returns:
            The complete response from the LLM

        """
        self._embed_frames(current_frame_index)
        messages = self._generate_messages(
            self.params.system_prompt,
            self.params.prompt,
            list(self._embedded_frames.values()),
        )

        answer = self._llm.create_chat_completion(
            messages=messages,
            temperature=self.params.temperature,
        )
        return answer

    def get_concatenated_frames(
        self, current_frame_index: int, axis: int = 0
    ) -> np.ndarray:
        """Concatenate frames at frame index

        Args:
            current_frame_index: Index of current frame

        Returns:
            The evaluated frames as a Numpy array

        """
        result = self._frames[
            current_frame_index - self.params.number_of_frames_to_evaluate
        ]
        for i in range(
            current_frame_index - self.params.number_of_frames_to_evaluate + 1,
            current_frame_index,
        ):
            result = np.concatenate((result, self._frames[i]), axis=axis)

        return result

    def _embed_frames(self, current_frame_index: int):
        """Embed video frames for LLM"""
        for j in range(
            current_frame_index - self.params.number_of_frames_to_evaluate,
            current_frame_index,
        ):
            if j in self._embedded_frames.keys():
                continue
            print(
                f"{self.video.name} - Embedding frame ({j + 1}/{len(self._frames)}) (Worker {self.i})"
            )
            _, buffer = cv2.imencode(".jpg", self._frames[j])
            data_array = array.array("B", buffer.tobytes())
            c_ubyte_ptr = (ctypes.c_ubyte * len(data_array)).from_buffer(data_array)
            with suppress_stdout_stderr(disable=self.params.verbose):
                self._embedded_frames[j] = (
                    self._llava_cpp.llava_image_embed_make_with_bytes(
                        ctx_clip=self._chat_handler.clip_ctx,
                        n_threads=self._llm.context_params.n_threads,
                        image_bytes=c_ubyte_ptr,
                        image_bytes_length=len(buffer),
                    )
                )

    def _generate_messages(self, system_prompt: str, prompt: str, images: list) -> list:
        """Generate the messages for the LLM

        Args:
            system_prompt: The system prompt to be passed to the LLM
            prompt: The prompt to be passed to the LLM
            images: The (embedded) images to be passed to the LLM

        Returns:
            Dictionary containing the messages

        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []},
        ]
        for i, image in enumerate(images):
            messages[1]["content"].append({"type": "text", "text": f"Image #{i}:"})
            messages[1]["content"].append(
                {
                    "type": "embedded_img",
                    "embedded_img": image,
                }
            )
        messages[1]["content"].append({"type": "text", "text": prompt})

        return messages
