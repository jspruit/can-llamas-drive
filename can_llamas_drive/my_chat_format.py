import ctypes
import json
from typing import Iterator, List, Optional, Union

import llama_cpp.llama as llama
import llama_cpp.llama_grammar as llama_grammar
import llama_cpp.llama_types as llama_types
from llama_cpp._utils import suppress_stdout_stderr
from llama_cpp.llama_chat_format import _convert_completion_to_chat, _get_system_message


class MyLlava15ChatHandler:
    _clip_free = None

    def __init__(self, _llava_cpp, clip_model_path: str, verbose: bool = False):
        self._llava_cpp = _llava_cpp
        self.clip_model_path = clip_model_path
        self.verbose = verbose
        self._clip_free = self._llava_cpp._libllava.clip_free  # type: ignore

        with suppress_stdout_stderr(disable=self.verbose):
            self.clip_ctx = self._llava_cpp.clip_model_load(
                self.clip_model_path.encode(), 0
            )

    def __del__(self):
        with suppress_stdout_stderr(disable=self.verbose):
            if self.clip_ctx is not None and self._clip_free is not None:
                self._clip_free(self.clip_ctx)
                self.clip_ctx = None

    def load_image(self, image_url: str) -> bytes:
        if image_url.startswith("data:"):
            import base64

            image_bytes = base64.b64decode(image_url.split(",")[1])
            return image_bytes
        else:
            import urllib.request

            with urllib.request.urlopen(image_url) as f:
                image_bytes = f.read()
                return image_bytes

    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        assert (
            llama.context_params.logits_all is True
        )  # BUG: logits_all=True is required for llava
        assert self.clip_ctx is not None
        system_prompt = _get_system_message(messages)
        system_prompt = (
            system_prompt
            if system_prompt != ""
            else "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions."
        )
        user_role = "\nUSER:"
        assistant_role = "\nASSISTANT:"
        llama.reset()
        llama.eval(llama.tokenize(system_prompt.encode("utf8"), add_bos=True))
        for message in messages:
            if message["role"] == "user" and message["content"] is not None:
                if isinstance(message["content"], str):
                    llama.eval(
                        llama.tokenize(
                            f"{user_role} {message['content']}".encode("utf8"),
                            add_bos=False,
                        )
                    )
                else:
                    assert isinstance(message["content"], list)
                    llama.eval(
                        llama.tokenize(f"{user_role} ".encode("utf8"), add_bos=False)
                    )
                    for content in message["content"]:
                        if content["type"] == "text":
                            llama.eval(
                                llama.tokenize(
                                    f"{content['text']}".encode("utf8"), add_bos=False
                                )
                            )
                        if content["type"] == "image_url":
                            image_bytes = (
                                self.load_image(content["image_url"]["url"])
                                if isinstance(content["image_url"], dict)
                                else self.load_image(content["image_url"])
                            )
                            import array

                            data_array = array.array("B", image_bytes)
                            c_ubyte_ptr = (
                                ctypes.c_ubyte * len(data_array)
                            ).from_buffer(data_array)
                            with suppress_stdout_stderr(disable=self.verbose):
                                embed = (
                                    self._llava_cpp.llava_image_embed_make_with_bytes(
                                        ctx_clip=self.clip_ctx,
                                        n_threads=llama.context_params.n_threads,
                                        image_bytes=c_ubyte_ptr,
                                        image_bytes_length=len(image_bytes),
                                    )
                                )
                            try:
                                n_past = ctypes.c_int(llama.n_tokens)
                                n_past_p = ctypes.pointer(n_past)
                                with suppress_stdout_stderr(disable=self.verbose):
                                    self._llava_cpp.llava_eval_image_embed(
                                        ctx_llama=llama.ctx,
                                        embed=embed,
                                        n_batch=llama.n_batch,
                                        n_past=n_past_p,
                                    )
                                assert llama.n_ctx() >= n_past.value
                                llama.n_tokens = n_past.value
                            finally:
                                with suppress_stdout_stderr(disable=self.verbose):
                                    self._llava_cpp.llava_image_embed_free(embed)
                        if content["type"] == "embedded_img":
                            n_past = ctypes.c_int(llama.n_tokens)
                            n_past_p = ctypes.pointer(n_past)
                            with suppress_stdout_stderr(disable=self.verbose):
                                self._llava_cpp.llava_eval_image_embed(
                                    ctx_llama=llama.ctx,
                                    embed=content["embedded_img"],
                                    n_batch=llama.n_batch,
                                    n_past=n_past_p,
                                )
                            assert llama.n_ctx() >= n_past.value
                            llama.n_tokens = n_past.value
            if message["role"] == "assistant" and message["content"] is not None:
                llama.eval(
                    llama.tokenize(
                        f"ASSISTANT: {message['content']}".encode("utf8"), add_bos=False
                    )
                )
                assert llama.n_ctx() >= llama.n_tokens
        llama.eval(llama.tokenize(f"{assistant_role}".encode("utf8"), add_bos=False))
        assert llama.n_ctx() >= llama.n_tokens

        prompt = llama.input_ids[: llama.n_tokens].tolist()

        if response_format is not None and response_format["type"] == "json_object":
            try:
                # create grammar from json schema
                if "schema" in response_format:
                    grammar = llama_grammar.LlamaGrammar.from_json_schema(
                        json.dumps(response_format["schema"])
                    )
            except Exception:
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF
                )

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            ),
            stream=stream,
        )
