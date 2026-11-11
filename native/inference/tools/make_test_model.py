#!/usr/bin/env python3
"""Write the tiny ONNX graphs the inference tests load.

Uses only the Python standard library. The graphs are SonicLab's, not
someone else's weights: one multiplies by two, and one repeats a multiply
by one so a background job stays busy long enough to cancel.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def field(number: int, wire: int, payload: bytes) -> bytes:
    return varint((number << 3) | wire) + payload


def length_delimited(number: int, payload: bytes) -> bytes:
    return field(number, 2, varint(len(payload)) + payload)


def proto_int(number: int, value: int) -> bytes:
    return field(number, 0, varint(value))


def proto_string(number: int, text: str) -> bytes:
    return length_delimited(number, text.encode())


def tensor(name: str, dims: list[int], values: list[float]) -> bytes:
    body = b""
    if dims:
        packed = b"".join(varint(dim) for dim in dims)
        body += length_delimited(1, packed)
    body += proto_int(2, 1)  # TensorProto.FLOAT
    raw = b"".join(struct.pack("<f", value) for value in values)
    body += length_delimited(4, raw)
    body += proto_string(8, name)
    return body


def value_info(name: str, dims: list[int]) -> bytes:
    shape = b"".join(length_delimited(1, proto_int(1, dim)) for dim in dims)
    tensor_type = proto_int(1, 1) + length_delimited(2, shape)
    type_proto = length_delimited(1, tensor_type)
    return proto_string(1, name) + length_delimited(2, type_proto)


def node(op: str, inputs: list[str], outputs: list[str], name: str) -> bytes:
    body = b"".join(proto_string(1, item) for item in inputs)
    body += b"".join(proto_string(2, item) for item in outputs)
    body += proto_string(3, name)
    body += proto_string(4, op)
    return body


def model(graph: bytes) -> bytes:
    opset = proto_int(2, 13)
    return proto_int(1, 8) + proto_string(2, "soniclab") + length_delimited(7, graph) + length_delimited(8, opset)


def scale_graph() -> bytes:
    body = length_delimited(1, node("Mul", ["x", "scale"], ["y"], "mul"))
    body += proto_string(2, "scale")
    body += length_delimited(5, tensor("scale", [1], [2.0]))
    body += length_delimited(11, value_info("x", [1, 4]))
    body += length_delimited(12, value_info("y", [1, 4]))
    return body


def chain_graph(steps: int) -> bytes:
    body = b""
    previous = "x"
    for index in range(steps):
        output = "y" if index + 1 == steps else f"t{index}"
        body += length_delimited(1, node("Mul", [previous, "one"], [output], f"mul{index}"))
        previous = output
    body += proto_string(2, "chain")
    body += length_delimited(5, tensor("one", [1], [1.0]))
    body += length_delimited(11, value_info("x", [1, 1048576]))
    body += length_delimited(12, value_info("y", [1, 1048576]))
    return body


def main() -> int:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "testdata"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "scale_mul.onnx").write_bytes(model(scale_graph()))
    (destination / "chain_mul.onnx").write_bytes(model(chain_graph(80)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
