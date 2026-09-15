"""Generic bounded HTTP (``bounded_http``): caller-supplied error type and
source label, streamed size cap, status and transport failures, request
forwarding, and JSON-object decoding. ``requests`` is mocked throughout."""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

import pytest
import requests

from tradingagents.dataflows import bounded_http


class _SourceError(Exception):
    pass


class _FakeResponse:
    def __init__(self, status_code=200, body=b"", headers=None, chunk=64 * 1024, fail_after=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body, self._chunk, self._fail_after = body, chunk, fail_after
        self.bytes_served = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, size):
        for i in range(0, len(self._body), self._chunk):
            if self._fail_after is not None and self.bytes_served >= self._fail_after:
                raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")
            piece = self._body[i:i + self._chunk]
            self.bytes_served += len(piece)
            yield piece


def _get(**kw):
    return bounded_http.bounded_request(
        "GET", "https://example.invalid/x", error_cls=_SourceError, source="SRC", accept="text/html", **kw)


@pytest.mark.unit
class BoundedRequestTests(unittest.TestCase):
    def _fails(self, msg_part, method="get", **patch_kw):
        with mock.patch.object(bounded_http.requests, method, **patch_kw), \
                self.assertRaises(_SourceError) as ctx:
            if method == "get":
                _get()
            else:
                bounded_http.bounded_request(
                    "POST", "https://example.invalid/x", error_cls=_SourceError, source="SRC",
                    accept="application/json", json_body={"a": 1})
        self.assertIn(msg_part, str(ctx.exception))
        return str(ctx.exception)

    def test_errors_use_caller_type_and_source_prefix(self):
        msg = self._fails("SRC fetch failed (Timeout)", side_effect=requests.Timeout("slow"))
        self.assertTrue(msg.endswith("for https://example.invalid/x"))
        self._fails("SRC fetch failed (SSLError)", side_effect=requests.exceptions.SSLError("cert"))
        self._fails("SRC fetch failed (ConnectionError)", method="post",
                    side_effect=requests.ConnectionError("reset"))

    def test_non_200_status(self):
        for status in (202, 403, 500):
            with self.subTest(status=status):
                self._fails(f"SRC fetch failed (HTTP {status})", return_value=_FakeResponse(status, b"x"))

    def test_stream_abandoned_at_cap_and_declared_length_refused(self):
        resp = _FakeResponse(body=b"x" * (bounded_http.DEFAULT_MAX_BODY_BYTES * 3))
        self._fails("SRC response exceeded", return_value=resp)
        self.assertLessEqual(resp.bytes_served,
                             bounded_http.DEFAULT_MAX_BODY_BYTES + bounded_http.CHUNK_BYTES)
        declared = _FakeResponse(body=b"ok", headers={"Content-Length": str(10**9)})
        self._fails("exceeded", return_value=declared)
        self.assertEqual(declared.bytes_served, 0)

    def test_custom_max_bytes(self):
        with mock.patch.object(bounded_http.requests, "get",
                               return_value=_FakeResponse(body=b"x" * 11, chunk=4)), \
                self.assertRaises(_SourceError):
            _get(max_bytes=10)
        with mock.patch.object(bounded_http.requests, "get",
                               return_value=_FakeResponse(body=b"x" * 10, chunk=4)):
            self.assertEqual(_get(max_bytes=10).body, b"x" * 10)

    def test_connection_dropped_mid_body(self):
        resp = _FakeResponse(body=b"x" * 300_000, chunk=64 * 1024, fail_after=64 * 1024)
        self._fails("SRC fetch failed (ChunkedEncodingError)", return_value=resp)

    def test_get_forwards_only_given_arguments(self):
        resp = _FakeResponse(body=b"ok", headers={"Content-Type": "text/csv; charset=MS950"})
        with mock.patch.object(bounded_http.requests, "get", return_value=resp) as get:
            out = _get(params={"date": "20260914"}, timeout=7.0, user_agent="UA/1")
        args, kwargs = get.call_args
        self.assertEqual(args, ("https://example.invalid/x",))
        self.assertEqual(kwargs, {"headers": {"User-Agent": "UA/1", "Accept": "text/html"},
                                  "timeout": 7.0, "stream": True, "params": {"date": "20260914"}})
        self.assertNotIn("verify", kwargs)
        self.assertEqual((out.url, out.status, out.body), ("https://example.invalid/x", 200, b"ok"))
        self.assertEqual(out.headers["content-type"], "text/csv; charset=MS950")   # case-insensitive

    def test_post_forwards_form_or_json_body(self):
        resp = _FakeResponse(body=b"{}")
        with mock.patch.object(bounded_http.requests, "post", return_value=resp) as post:
            bounded_http.bounded_request("POST", "https://example.invalid/f", error_cls=_SourceError,
                                         source="SRC", accept="text/csv", data={"type": "Daily"})
        self.assertEqual(post.call_args.kwargs["data"], {"type": "Daily"})
        self.assertNotIn("json", post.call_args.kwargs)
        resp = _FakeResponse(body=b"{}")
        with mock.patch.object(bounded_http.requests, "post", return_value=resp) as post:
            bounded_http.bounded_request("POST", "https://example.invalid/j", error_cls=_SourceError,
                                         source="SRC", accept="application/json", json_body={"a": 1})
        self.assertEqual(post.call_args.kwargs["json"], {"a": 1})
        self.assertNotIn("data", post.call_args.kwargs)
        self.assertEqual(post.call_args.kwargs["timeout"], bounded_http.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(post.call_args.kwargs["headers"]["User-Agent"], bounded_http.DEFAULT_USER_AGENT)

    def test_unsupported_method(self):
        with self.assertRaises(ValueError):
            bounded_http.bounded_request("PUT", "https://example.invalid/x", error_cls=_SourceError,
                                         source="SRC", accept="*/*")


@pytest.mark.unit
class DecodeJsonObjectTests(unittest.TestCase):
    def _resp(self, body):
        return bounded_http.BoundedResponse(url="https://example.invalid/j", status=200,
                                            headers={}, body=body)

    def test_object_decodes(self):
        self.assertEqual(bounded_http.decode_json_object(
            self._resp('{"k": "台積電"}'.encode()), error_cls=_SourceError, source="SRC"), {"k": "台積電"})

    def test_malformed_and_non_object(self):
        for body, part in ((b"<html>busy</html>", "SRC returned malformed JSON"),
                           (b"[1, 2]", "SRC returned an unexpected JSON shape"),
                           (b"\xff\xfe", "SRC returned malformed JSON")):
            with self.subTest(body=body), self.assertRaises(_SourceError) as ctx:
                bounded_http.decode_json_object(self._resp(body), error_cls=_SourceError, source="SRC")
            self.assertIn(part, str(ctx.exception))


@pytest.mark.unit
def test_module_names_no_data_source():
    source = inspect.getsource(bounded_http).lower()
    for name in ("mops", "twse", "tpex"):
        assert name not in source, name
