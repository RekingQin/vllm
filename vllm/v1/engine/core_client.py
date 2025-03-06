# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import queue
import signal
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Callable, Optional, Union

import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import (get_open_zmq_inproc_path, get_open_zmq_ipc_path,
                        kill_process_tree, make_zmq_socket)
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import BackgroundProcHandle

logger = init_logger(__name__)

AnyFuture = Union[asyncio.Future[Any], Future[Any]]


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing 
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported.")

        if multiprocess_mode and asyncio_mode:
            if vllm_config.parallel_config.data_parallel_size > 1:
                return DPAsyncMPClient(vllm_config, executor_class, log_stats)
            else:
                return AsyncMPClient(vllm_config, executor_class, log_stats)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @abstractmethod
    def shutdown(self):
        ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True) -> None:
        raise NotImplementedError

    def reset_prefix_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    def wake_up(self) -> None:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(self, is_start: bool = True) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up_async(self) -> None:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended 
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True) -> None:
        self.engine_core.profile(is_start)

    def reset_prefix_cache(self) -> None:
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1) -> None:
        self.engine_core.sleep(level)

    def wake_up(self) -> None:
        self.engine_core.wake_up()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)


class CoreEngine:
    """One per data parallel rank."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ctx: Union[zmq.Context, zmq.asyncio.Context],
        output_path: str,
        index: int = 0,
    ):
        # Paths and sockets for IPC.
        input_path = get_open_zmq_ipc_path()
        self.input_socket = make_zmq_socket(ctx, input_path,
                                            zmq.constants.PUSH)
        try:
            # Start EngineCore in background process.
            self.proc_handle = BackgroundProcHandle(
                input_path=input_path,
                output_path=output_path,
                process_name=f"EngineCore_{index}",
                target_fn=EngineCoreProc.run_engine_core,
                process_kwargs={
                    "vllm_config": vllm_config,
                    "dp_rank": index,
                    "executor_class": executor_class,
                    "log_stats": log_stats,
                })

            self.num_reqs_in_flight = 0
        finally:
            if not hasattr(self, "num_reqs_in_flight"):
                # Ensure socket is closed if process fails to start.
                self.close()

    def send_multipart(self, msg_parts: Sequence):
        return self.input_socket.send_multipart(msg_parts, copy=False)

    def close(self):
        if proc_handle := getattr(self, "proc_handle", None):
            proc_handle.shutdown()
        if socket := getattr(self, "input_socket", None):
            socket.close(linger=0)


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    core_engines: list[CoreEngine] = field(default_factory=list)
    ctx: Union[zmq.Context] = None
    output_socket: Union[zmq.Socket, zmq.asyncio.Socket] = None
    shutdown_path: Optional[str] = None

    def __call__(self):
        """Clean up background resources."""

        for core_engine in self.core_engines:
            core_engine.close()

        # ZMQ context termination can hang if the sockets
        # aren't explicitly closed first.
        if self.output_socket is not None:
            self.output_socket.close(linger=0)
        if self.shutdown_path is not None:
            # We must ensure that the sync output socket is
            # closed cleanly in its own thread.
            with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                shutdown_sender.connect(self.shutdown_path)
                # Send shutdown signal.
                shutdown_sender.send(b'')


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    
        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen. We kill the process tree here so that the
        # stack trace is very evident.
        # TODO(rob): rather than killing the main process, we should
        # figure out how to raise an AsyncEngineDeadError and
        # handle at the API server level so we can return a better
        # error code to the clients calling VLLM.
        def sigusr1_handler(signum, frame):
            logger.fatal("Got fatal signal from worker processes, shutting "
                         "down. See stack trace above for root cause issue.")
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        # Serialization setup.
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup.
        sync_ctx = zmq.Context()
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected,  even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)

        # Paths and sockets for IPC.
        self.output_path = get_open_zmq_ipc_path()

        new_core_engine = lambda index: CoreEngine(
            vllm_config, executor_class, log_stats, self.ctx, self.output_path,
            index)

        # Start engine core process(es).
        self._init_core_engines(vllm_config, new_core_engine,
                                self.resources.core_engines)

        # Wait for engine core process(es) to start.
        for engine in self.resources.core_engines:
            engine.proc_handle.wait_for_startup()

        self.utility_results: dict[int, AnyFuture] = {}

    def _init_core_engines(
        self,
        vllm_config: VllmConfig,
        new_core_engine: Callable[[int], CoreEngine],
        core_engines: list[CoreEngine],
    ) -> None:

        # Default case - single core engine.
        core_engine = new_core_engine(0)
        core_engines.append(core_engine)
        self.core_engine = core_engine

    def shutdown(self):
        self._finalizer()


def _process_utility_output(output: UtilityOutput,
                            utility_results: dict[int, AnyFuture]):
    """Set the result from a utility method in the waiting future"""
    future = utility_results.pop(output.call_id)
    if output.failure_message is not None:
        future.set_exception(Exception(output.failure_message))
    else:
        future.set_result(output.result)


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.outputs_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        output_path = self.output_path
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        self.resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            shutdown_socket = ctx.socket(zmq.PAIR)
            shutdown_socket.bind(shutdown_path)
            out_socket = make_zmq_socket(ctx, output_path, zmq.constants.PULL)
            try:
                poller = zmq.Poller()
                poller.register(shutdown_socket)
                poller.register(out_socket)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frame = out_socket.recv(copy=False)
                    outputs = decoder.decode(frame.buffer)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output,
                                                utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(target=process_outputs_socket,
                                          name="EngineCoreOutputQueueThread",
                                          daemon=True)
        self.output_queue_thread.start()

    def get_output(self) -> EngineCoreOutputs:
        return self.outputs_queue.get()

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        # (RequestType, SerializedRequest)
        msg = (request_type.value, self.encoder.encode(request))
        self.core_engine.send_multipart(msg)

    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY,
                         (call_id, method, args))

        return future.result()

    def add_request(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True) -> None:
        self.call_utility("profile", is_start)

    def reset_prefix_cache(self) -> None:
        self.call_utility("reset_prefix_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1) -> None:
        self.call_utility("sleep", level)

    def wake_up(self) -> None:
        self.call_utility("wake_up")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.outputs_queue: Optional[asyncio.Queue[EngineCoreOutputs]] = None
        self.queue_task: Optional[asyncio.Task] = None

        self.output_processor: Optional[Callable[
            [AsyncMPClient, EngineCoreOutputs], Awaitable[None]]] = None

    async def _start_output_queue_task(self):
        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        self.outputs_queue = asyncio.Queue()
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_processor = self.output_processor
        _self_ref = weakref.ref(self) if output_processor else None
        output_path = self.output_path
        output_socket = make_zmq_socket(self.ctx, output_path,
                                        zmq.constants.PULL)
        self.resources.output_socket = output_socket

        async def process_outputs_socket():
            while True:
                (frame, ) = await output_socket.recv_multipart(copy=False)
                outputs: EngineCoreOutputs = decoder.decode(frame.buffer)
                if outputs.utility_output:
                    _process_utility_output(outputs.utility_output,
                                            utility_results)
                    continue

                if output_processor is not None:
                    assert _self_ref is not None
                    _self = _self_ref()
                    if not _self:
                        return
                    await output_processor(_self, outputs)

                if outputs.outputs or outputs.scheduler_stats:
                    outputs_queue.put_nowait(outputs)

        self.queue_task = asyncio.create_task(process_outputs_socket(),
                                              name="EngineCoreOutputQueueTask")

    async def get_output_async(self) -> EngineCoreOutputs:
        if self.outputs_queue is None:
            await self._start_output_queue_task()
            assert self.outputs_queue is not None
        return await self.outputs_queue.get()

    async def _send_input(self, request_type: EngineCoreRequestType,
                          request: Any) -> None:
        await self.core_engine.send_multipart(
            (request_type.value, self.encoder.encode(request)))

        if self.outputs_queue is None:
            await self._start_output_queue_task()

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method,
                                              *args,
                                              engine=self.core_engine)

    async def _call_utility_async(
        self,
        method: str,
        *args,
        engine: CoreEngine,
    ) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        await engine.send_multipart(
            (EngineCoreRequestType.UTILITY.value, (call_id, method, args)))
        return await future

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None
        await self._send_input(EngineCoreRequestType.ADD, request)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self.call_utility_async("profile", is_start)

    async def reset_prefix_cache_async(self) -> None:
        await self.call_utility_async("reset_prefix_cache")

    async def sleep_async(self, level: int = 1) -> None:
        await self.call_utility_async("sleep", level)

    async def wake_up_async(self) -> None:
        await self.call_utility_async("wake_up")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):
        super().__init__(vllm_config, executor_class, log_stats)

        assert len(self.core_engines) > 1

        # Control message used for triggering dp idle mode loop.
        self.start_dp_msg = (EngineCoreRequestType.START_DP.value,
                             self.encoder.encode(None))

        self.num_engines_running = 0
        self.reqs_in_flight: dict[str, CoreEngine] = {}

        self.output_processor = DPAsyncMPClient.process_engine_outputs  # type: ignore[assignment]
        self.outputs_counter = 0

    def _init_core_engines(
        self,
        vllm_config: VllmConfig,
        new_core_engine: Callable[[int], CoreEngine],
        core_engines: list[CoreEngine],
    ) -> None:

        # Launch a core engine for each data parallel rank.
        dp_size = vllm_config.parallel_config.data_parallel_size
        for i in range(dp_size):
            core_engines.append(new_core_engine(i))

        self.core_engines = core_engines

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (await asyncio.gather(*[
            self._call_utility_async(method, *args, engine=engine)
            for engine in self.core_engines
        ]))[0]

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None

        if self.outputs_queue is None:
            await self._start_output_queue_task()

        msg = (EngineCoreRequestType.ADD.value, self.encoder.encode(request))

        chosen_engine = self.get_core_engine_for_request()
        self.reqs_in_flight[request.request_id] = chosen_engine
        chosen_engine.num_reqs_in_flight += 1
        if self.num_engines_running >= len(self.core_engines):
            await chosen_engine.send_multipart(msg)
        else:
            # Send request to chosen engine and dp start loop
            # control message to all other engines.
            self.num_engines_running += len(self.core_engines)
            await asyncio.gather(*[
                engine.send_multipart(msg if engine is
                                      chosen_engine else self.start_dp_msg)
                for engine in self.core_engines
            ])

    def get_core_engine_for_request(self) -> CoreEngine:
        return min(self.core_engines, key=lambda e: e.num_reqs_in_flight)

    @staticmethod
    async def process_engine_outputs(self: "DPAsyncMPClient",
                                     outputs: EngineCoreOutputs):
        if self.reqs_in_flight:
            for req_id in outputs.finished_requests or ():
                if engine := self.reqs_in_flight.pop(req_id, None):
                    engine.num_reqs_in_flight -= 1

        if outputs.engine_paused:
            assert self.num_engines_running >= 1
            self.num_engines_running -= 1
            if self.num_engines_running == 0 and self.reqs_in_flight:
                # If there are requests in flight here, they must have
                # been sent after the engines paused. We must make
                # sure to start the other engines:
                self.num_engines_running = len(self.core_engines)
                coros = [
                    engine.send_multipart(self.start_dp_msg)
                    for engine in self.core_engines
                    if not engine.num_reqs_in_flight
                ]
                if coros:
                    await asyncio.gather(*coros)

        if outputs.scheduler_stats:
            # Unset the final flag if we haven't yet received outputs
            # from all engines for this step.
            self.outputs_counter += 1
            if self.outputs_counter == len(self.core_engines):
                self.outputs_counter = 0
            else:
                outputs.final_outputs_for_step = False

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine: dict[CoreEngine, list[str]] = {}
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine.setdefault(engine, []).append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(self, request_ids: list[str],
                              engine: CoreEngine) -> None:
        await engine.send_multipart((EngineCoreRequestType.ABORT.value,
                                     self.encoder.encode(request_ids)))
