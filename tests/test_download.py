from __future__ import annotations

import http.server
import json
import threading
import zipfile

import pytest

from src.data.download import (
    DataPaths,
    DownloadError,
    _describe,
    _sha256,
    extract_member_over_http,
    image_ids_for_qids,
    load_air_d_qids,
    normalise_air_d_layout,
    verify,
)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Minimal static server that honours single-range requests."""

    payload: bytes = b""

    def log_message(self, *args):  # noqa: ANN002
        pass

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if rng is None:
            body, status = self.payload, 200
        else:
            start, _, end = rng.removeprefix("bytes=").partition("-")
            lo = int(start)
            hi = int(end) if end else len(self.payload) - 1
            body, status = self.payload[lo : hi + 1], 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def range_server(tmp_path):
    """Serve a zip archive over HTTP with byte-range support."""
    zip_path = tmp_path / "questions.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("filler.bin", b"x" * 500_000)
        zf.writestr(
            "val_balanced_questions.json",
            json.dumps({"q1": {"imageId": "2317911", "question": "?", "answer": "yes"}}),
        )
        zf.writestr("other/train_questions.json", json.dumps({"q9": {}}))
    _RangeHandler.payload = zip_path.read_bytes()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/questions.zip"
    server.shutdown()
    server.server_close()


def test_extract_member_over_http(range_server, tmp_path):
    dest = tmp_path / "out" / "val_balanced_questions.json"
    extract_member_over_http(range_server, "val_balanced_questions.json", dest)
    data = json.loads(dest.read_text())
    assert data["q1"]["imageId"] == "2317911"


def test_extract_member_matches_on_basename(range_server, tmp_path):
    dest = tmp_path / "train.json"
    extract_member_over_http(range_server, "train_questions.json", dest)
    assert json.loads(dest.read_text()) == {"q9": {}}


def test_extract_missing_member_raises(range_server, tmp_path):
    with pytest.raises(DownloadError, match="not found"):
        extract_member_over_http(range_server, "nope.json", tmp_path / "x.json")


def _make_air_d_tree(root):
    """Build an unpacked AiR-D tree mirroring the real Drive archives.

    The maps archive unpacks to ``AiR-D/{aggregated_maps,temporal_maps}``, the
    scanpath archive to ``air-fixations/{fix/<qid>/<subject>.mat,
    consolidated_answers.json}``.
    """
    air_d = root / "raw" / "air_d"
    for name in ("fixmaps", "fixmaps_corr", "fixmaps_incorr"):
        (air_d / "AiR-D" / "aggregated_maps" / name).mkdir(parents=True)
        for qid in ("001", "002"):
            (air_d / "AiR-D" / "aggregated_maps" / name / f"{qid}.png").write_bytes(b"\x89PNG")
    for name in ("fixmaps_0s", "fixmaps_1s", "fixmaps_2s"):
        (air_d / "AiR-D" / "temporal_maps" / name).mkdir(parents=True)
        for qid in ("001", "002"):
            (air_d / "AiR-D" / "temporal_maps" / name / f"{qid}.png").write_bytes(b"\x89PNG")
    for qid in ("001", "002"):
        (air_d / "air-fixations" / "fix" / qid).mkdir(parents=True)
        for subject in ("AL", "YN"):
            (air_d / "air-fixations" / "fix" / qid / f"{subject}.mat").write_bytes(b"\x00")
    (air_d / "air-fixations" / "consolidated_answers.json").write_text(
        json.dumps(
            {
                "accuracy": {"001": 1.0, "002": 0.5},
                "question": {"001": "a?", "002": "b?"},
                "answer": {"001": "yes", "002": "no"},
                "AL": {"001": "yes", "002": "no"},
                "YN": {"001": "yes", "002": "yes"},
            }
        )
    )


def test_normalise_air_d_layout(tmp_path):
    _make_air_d_tree(tmp_path)
    paths = DataPaths(tmp_path)
    normalise_air_d_layout(paths)

    assert (paths.aggregated_maps / "fixmaps" / "001.png").exists()
    assert (paths.aggregated_maps / "fixmaps_incorr" / "002.png").exists()
    assert (paths.temporal_maps / "fixmaps_1s" / "001.png").exists()
    assert (paths.scanpaths / "fix" / "001" / "AL.mat").exists()
    assert (paths.scanpaths / "consolidated_answers.json").exists()
    # The archive-specific wrappers are gone once everything has been moved out.
    assert not (paths.air_d / "AiR-D").exists()
    assert not (paths.air_d / "air-fixations").exists()


def test_normalise_air_d_layout_is_idempotent(tmp_path):
    _make_air_d_tree(tmp_path)
    paths = DataPaths(tmp_path)
    normalise_air_d_layout(paths)
    normalise_air_d_layout(paths)
    assert sorted(p.name for p in paths.aggregated_maps.iterdir()) == [
        "fixmaps",
        "fixmaps_corr",
        "fixmaps_incorr",
    ]


def test_normalise_fails_loudly_on_empty_tree(tmp_path):
    paths = DataPaths(tmp_path)
    paths.mkdirs()
    with pytest.raises(DownloadError, match="fixmaps"):
        normalise_air_d_layout(paths)


def test_load_air_d_qids(tmp_path):
    _make_air_d_tree(tmp_path)
    paths = DataPaths(tmp_path)
    normalise_air_d_layout(paths)
    assert load_air_d_qids(paths) == ["001", "002"]


def test_load_air_d_qids_drops_maps_without_temporal_bins(tmp_path):
    """The real release ships 1,454 aggregated maps for 1,422 usable questions."""
    _make_air_d_tree(tmp_path)
    paths = DataPaths(tmp_path)
    normalise_air_d_layout(paths)
    (paths.aggregated_maps / "fixmaps" / "003.png").write_bytes(b"\x89PNG")
    assert load_air_d_qids(paths) == ["001", "002"]


def test_image_ids_for_qids_reports_missing():
    questions = {"001": {"imageId": "111"}, "002": {"imageId": "111"}}
    image_ids, missing = image_ids_for_qids(questions, ["001", "002", "003"])
    assert image_ids == ["111"]
    assert missing == ["003"]


def test_verify_reports_missing_images(tmp_path):
    _make_air_d_tree(tmp_path)
    paths = DataPaths(tmp_path)
    normalise_air_d_layout(paths)
    paths.gqa.mkdir(parents=True, exist_ok=True)
    paths.questions_json.write_text(
        json.dumps({"001": {"imageId": "111"}, "002": {"imageId": "222"}})
    )
    paths.images.mkdir(parents=True, exist_ok=True)
    (paths.images / "111.jpg").write_bytes(b"\xff\xd8")

    problems = verify(tmp_path)
    assert any("1 images missing" in p for p in problems)
    assert verify(tmp_path, require_images=False) == []


def test_manifest_entries(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}")
    d = tmp_path / "sub"
    (d / "deep").mkdir(parents=True)
    (d / "deep" / "x.png").write_bytes(b"1234")

    file_entry = _describe(f, tmp_path)
    assert file_entry.kind == "file"
    assert file_entry.sha256 == _sha256(f)

    dir_entry = _describe(d, tmp_path)
    assert dir_entry.kind == "dir"
    assert dir_entry.count == 1
    assert dir_entry.bytes == 4
