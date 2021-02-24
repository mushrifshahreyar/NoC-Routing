"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: special_math_ops.cc
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export


def dawsn(x, name=None):
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "Dawsn", name,
        tld.op_callbacks, x)
      return _result
    except _core._FallbackException:
      try:
        return dawsn_eager_fallback(
            x, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Dawsn", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Dawsn", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Dawsn = tf_export("raw_ops.Dawsn")(_ops.to_raw_op(dawsn))


def dawsn_eager_fallback(x, name, ctx):
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Dawsn", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Dawsn", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def expint(x, name=None):
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "Expint", name,
        tld.op_callbacks, x)
      return _result
    except _core._FallbackException:
      try:
        return expint_eager_fallback(
            x, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Expint", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Expint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Expint = tf_export("raw_ops.Expint")(_ops.to_raw_op(expint))


def expint_eager_fallback(x, name, ctx):
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Expint", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Expint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def fresnel_cos(x, name=None):
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "FresnelCos", name,
        tld.op_callbacks, x)
      return _result
    except _core._FallbackException:
      try:
        return fresnel_cos_eager_fallback(
            x, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FresnelCos", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FresnelCos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FresnelCos = tf_export("raw_ops.FresnelCos")(_ops.to_raw_op(fresnel_cos))


def fresnel_cos_eager_fallback(x, name, ctx):
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"FresnelCos", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FresnelCos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def fresnel_sin(x, name=None):
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "FresnelSin", name,
        tld.op_callbacks, x)
      return _result
    except _core._FallbackException:
      try:
        return fresnel_sin_eager_fallback(
            x, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FresnelSin", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FresnelSin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FresnelSin = tf_export("raw_ops.FresnelSin")(_ops.to_raw_op(fresnel_sin))


def fresnel_sin_eager_fallback(x, name, ctx):
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"FresnelSin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FresnelSin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def spence(x, name=None):
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "Spence", name,
        tld.op_callbacks, x)
      return _result
    except _core._FallbackException:
      try:
        return spence_eager_fallback(
            x, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Spence", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Spence", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Spence = tf_export("raw_ops.Spence")(_ops.to_raw_op(spence))


def spence_eager_fallback(x, name, ctx):
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Spence", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Spence", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

