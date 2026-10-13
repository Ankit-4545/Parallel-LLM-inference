from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ItemStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[ItemStatus]
    STALE_EPOCH: _ClassVar[ItemStatus]
    FUTURE_EPOCH: _ClassVar[ItemStatus]
    STALE_STEP: _ClassVar[ItemStatus]
    OUT_OF_ORDER: _ClassVar[ItemStatus]
    STALE_ATTEMPT: _ClassVar[ItemStatus]
    UNKNOWN_REQUEST: _ClassVar[ItemStatus]
    REQUEST_CLOSED: _ClassVar[ItemStatus]
    CACHE_FULL: _ClassVar[ItemStatus]
    QUEUE_FULL: _ClassVar[ItemStatus]
    DOWNSTREAM_UNAVAILABLE: _ClassVar[ItemStatus]
    INTERNAL_ERROR: _ClassVar[ItemStatus]

class RequestStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUNNING: _ClassVar[RequestStatus]
    COMPLETED: _ClassVar[RequestStatus]
    OVERLOADED: _ClassVar[RequestStatus]
    TIMEOUT: _ClassVar[RequestStatus]
    CANCELLED: _ClassVar[RequestStatus]
    FAILED: _ClassVar[RequestStatus]
    INVALID: _ClassVar[RequestStatus]
OK: ItemStatus
STALE_EPOCH: ItemStatus
FUTURE_EPOCH: ItemStatus
STALE_STEP: ItemStatus
OUT_OF_ORDER: ItemStatus
STALE_ATTEMPT: ItemStatus
UNKNOWN_REQUEST: ItemStatus
REQUEST_CLOSED: ItemStatus
CACHE_FULL: ItemStatus
QUEUE_FULL: ItemStatus
DOWNSTREAM_UNAVAILABLE: ItemStatus
INTERNAL_ERROR: ItemStatus
RUNNING: RequestStatus
COMPLETED: RequestStatus
OVERLOADED: RequestStatus
TIMEOUT: RequestStatus
CANCELLED: RequestStatus
FAILED: RequestStatus
INVALID: RequestStatus

class Tensor(_message.Message):
    __slots__ = ("shape", "dtype", "data")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    data: bytes
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class StageItem(_message.Message):
    __slots__ = ("request_id", "step", "attempt", "reset", "use_cache", "release_after", "token_ids", "hidden")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_AFTER_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step: int
    attempt: int
    reset: bool
    use_cache: bool
    release_after: bool
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    hidden: Tensor
    def __init__(self, request_id: _Optional[str] = ..., step: _Optional[int] = ..., attempt: _Optional[int] = ..., reset: _Optional[bool] = ..., use_cache: _Optional[bool] = ..., release_after: _Optional[bool] = ..., token_ids: _Optional[_Iterable[int]] = ..., hidden: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class ItemResult(_message.Message):
    __slots__ = ("request_id", "step", "attempt", "status", "token_id", "hidden", "replayed", "detail")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_FIELD_NUMBER: _ClassVar[int]
    REPLAYED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step: int
    attempt: int
    status: ItemStatus
    token_id: int
    hidden: Tensor
    replayed: bool
    detail: str
    def __init__(self, request_id: _Optional[str] = ..., step: _Optional[int] = ..., attempt: _Optional[int] = ..., status: _Optional[_Union[ItemStatus, str]] = ..., token_id: _Optional[int] = ..., hidden: _Optional[_Union[Tensor, _Mapping]] = ..., replayed: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...

class StageTiming(_message.Message):
    __slots__ = ("stage", "queue_ms", "compute_ms", "deserialize_ms", "serialize_ms", "downstream_ms", "bytes_in", "bytes_out", "link_delay_ms")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_MS_FIELD_NUMBER: _ClassVar[int]
    DESERIALIZE_MS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZE_MS_FIELD_NUMBER: _ClassVar[int]
    DOWNSTREAM_MS_FIELD_NUMBER: _ClassVar[int]
    BYTES_IN_FIELD_NUMBER: _ClassVar[int]
    BYTES_OUT_FIELD_NUMBER: _ClassVar[int]
    LINK_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    stage: int
    queue_ms: float
    compute_ms: float
    deserialize_ms: float
    serialize_ms: float
    downstream_ms: float
    bytes_in: int
    bytes_out: int
    link_delay_ms: float
    def __init__(self, stage: _Optional[int] = ..., queue_ms: _Optional[float] = ..., compute_ms: _Optional[float] = ..., deserialize_ms: _Optional[float] = ..., serialize_ms: _Optional[float] = ..., downstream_ms: _Optional[float] = ..., bytes_in: _Optional[int] = ..., bytes_out: _Optional[int] = ..., link_delay_ms: _Optional[float] = ...) -> None: ...

class ForwardRequest(_message.Message):
    __slots__ = ("stage_epochs", "batch_id", "items")
    STAGE_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    stage_epochs: _containers.RepeatedScalarFieldContainer[int]
    batch_id: int
    items: _containers.RepeatedCompositeFieldContainer[StageItem]
    def __init__(self, stage_epochs: _Optional[_Iterable[int]] = ..., batch_id: _Optional[int] = ..., items: _Optional[_Iterable[_Union[StageItem, _Mapping]]] = ...) -> None: ...

class ForwardResponse(_message.Message):
    __slots__ = ("results", "timings", "worker_epoch")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TIMINGS_FIELD_NUMBER: _ClassVar[int]
    WORKER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[ItemResult]
    timings: _containers.RepeatedCompositeFieldContainer[StageTiming]
    worker_epoch: int
    def __init__(self, results: _Optional[_Iterable[_Union[ItemResult, _Mapping]]] = ..., timings: _Optional[_Iterable[_Union[StageTiming, _Mapping]]] = ..., worker_epoch: _Optional[int] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("stage", "epoch", "pid", "queue_depth", "cached_requests", "ready")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    CACHED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    stage: int
    epoch: int
    pid: int
    queue_depth: int
    cached_requests: int
    ready: bool
    def __init__(self, stage: _Optional[int] = ..., epoch: _Optional[int] = ..., pid: _Optional[int] = ..., queue_depth: _Optional[int] = ..., cached_requests: _Optional[int] = ..., ready: _Optional[bool] = ...) -> None: ...

class ReleaseRequest(_message.Message):
    __slots__ = ("request_ids",)
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, request_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ReleaseResponse(_message.Message):
    __slots__ = ("released",)
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    released: int
    def __init__(self, released: _Optional[int] = ...) -> None: ...

class JsonRequest(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: str
    def __init__(self, json: _Optional[str] = ...) -> None: ...

class JsonResponse(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: str
    def __init__(self, json: _Optional[str] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "prompt", "prompt_token_ids", "max_new_tokens", "timeout_s")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: str
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    max_new_tokens: int
    timeout_s: float
    def __init__(self, request_id: _Optional[str] = ..., prompt: _Optional[str] = ..., prompt_token_ids: _Optional[_Iterable[int]] = ..., max_new_tokens: _Optional[int] = ..., timeout_s: _Optional[float] = ...) -> None: ...

class GenerateChunk(_message.Message):
    __slots__ = ("request_id", "step", "token_id", "text", "status", "final", "detail", "ttft_ms", "total_ms", "token_ids")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    TTFT_MS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step: int
    token_id: int
    text: str
    status: RequestStatus
    final: bool
    detail: str
    ttft_ms: float
    total_ms: float
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, request_id: _Optional[str] = ..., step: _Optional[int] = ..., token_id: _Optional[int] = ..., text: _Optional[str] = ..., status: _Optional[_Union[RequestStatus, str]] = ..., final: _Optional[bool] = ..., detail: _Optional[str] = ..., ttft_ms: _Optional[float] = ..., total_ms: _Optional[float] = ..., token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CancelRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class CancelResponse(_message.Message):
    __slots__ = ("cancelled",)
    CANCELLED_FIELD_NUMBER: _ClassVar[int]
    cancelled: bool
    def __init__(self, cancelled: _Optional[bool] = ...) -> None: ...
