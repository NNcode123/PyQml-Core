from __future__ import annotations
import numpy
import typing
__all__: list[str] = ['Tensor', 'arange', 'dtype', 'einsum', 'float32', 'float64', 'int16', 'int32', 'int64', 'int8', 'to_numpy', 'uint8']
class Tensor:
    def __add__(self, arg0: Tensor) -> Tensor:
        ...
    def __init__(self, data: list, dim: list, type: typing.Any = None) -> None:
        ...
    def __mul__(self, arg0: Tensor) -> Tensor:
        ...
    def __repr__(self) -> str:
        ...
    def __sub__(self, arg0: Tensor) -> Tensor:
        ...
    def __truediv__(self, arg0: Tensor) -> Tensor:
        ...
    def astype(self, dtype: typing.Any, copy: bool = False) -> Tensor:
        ...
    @property
    def dtype(self) -> DType:
        ...
    @property
    def shape(self) -> list[int]:
        ...
class dtype:
    """
    Members:
    
      uint8
    
      int8
    
      int16
    
      int32
    
      int64
    
      float32
    
      float64
    """
    __members__: typing.ClassVar[dict[str, dtype]]  # value = {'uint8': <dtype.uint8: 0>, 'int8': <dtype.int8: 1>, 'int16': <dtype.int16: 2>, 'int32': <dtype.int32: 3>, 'int64': <dtype.int64: 4>, 'float32': <dtype.float32: 5>, 'float64': <dtype.float64: 6>}
    float32: typing.ClassVar[dtype]  # value = <dtype.float32: 5>
    float64: typing.ClassVar[dtype]  # value = <dtype.float64: 6>
    int16: typing.ClassVar[dtype]  # value = <dtype.int16: 2>
    int32: typing.ClassVar[dtype]  # value = <dtype.int32: 3>
    int64: typing.ClassVar[dtype]  # value = <dtype.int64: 4>
    int8: typing.ClassVar[dtype]  # value = <dtype.int8: 1>
    uint8: typing.ClassVar[dtype]  # value = <dtype.uint8: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self: numpy.dtype[typing.Any]) -> int:
        ...
    def __init__(self: numpy.dtype[typing.Any], value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self: numpy.dtype[typing.Any]) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self: numpy.dtype[typing.Any]) -> str:
        ...
    def __setstate__(self: numpy.dtype[typing.Any], state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def arange(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> Tensor:
    ...
def einsum(arg0: Tensor, arg1: Tensor, arg2: typing.Any, arg3: typing.Any) -> Tensor:
    ...
def to_numpy(arg0: Tensor) -> numpy.ndarray:
    ...
float32: dtype  # value = <dtype.float32: 5>
float64: dtype  # value = <dtype.float64: 6>
int16: dtype  # value = <dtype.int16: 2>
int32: dtype  # value = <dtype.int32: 3>
int64: dtype  # value = <dtype.int64: 4>
int8: dtype  # value = <dtype.int8: 1>
uint8: dtype  # value = <dtype.uint8: 0>
