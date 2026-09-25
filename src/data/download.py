"""Acquisition of the AiR-D eye-tracking data, the GQA questions and the images.

Three sources, three very different access patterns:

* **AiR-D** ships as Google Drive archives (aggregated/temporal saliency maps and
  raw scanpaths). Fetched with ``gdown``.
* **GQA questions** live inside a 1.5 GB zip of which we need a single member.
  The server supports byte ranges, so the zip is read remotely and only
  ``val_balanced_questions.json`` is extracted.
* **Images** are Visual Genome files; GQA image IDs are VG image IDs. The GQA
  image archive is 21.8 GB for the 987 images we need, so images are fetched
  individually from ``VG_100K`` with a ``VG_100K_2`` fallback.

Everything is verified after the fact and the module raises :class:`DownloadError`
rather than leaving a half-populated data directory behind.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# --- Source constants -------------------------------------------------------

AIR_D_MAPS_DRIVE_ID = "1_yWlv3GXYw0-qBan5pPEmV8vd-Y8Duh5"
AIR_D_SCANPATHS_DRIVE_ID = "1HFlX09kRB2lOa6qYihFjQ0Q8KSvxfPcZ"
AIR_D_HOLDOUT_DRIVE_IDS = {
    "general_split": "1xWpFjMyXGvIKrM53S4AjAd0qla8eWKE2",
    "correctness_split": "1pM-N6ewuQz-4BXaMvKpkJ_GcX4WO_q_e",
}

GQA_QUESTIONS_ZIP_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_QUESTIONS_MEMBER = "val_balanced_questions.json"

VG_IMAGE_URLS = (
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
)

# Published dataset size (idea.md 4.2). Used as a sanity check, not a hard gate:
# the authors may re-release with small differences.
AIR_D_EXPECTED_QUESTIONS = 1422
AIR_D_EXPECTED_IMAGES = 987

# Directory names inside the AiR-D archives that we care about.
AGGREGATED_MAP_DIRS = ("fixmaps", "fixmaps_corr", "fixmaps_incorr")
SCANPATH_DIR = "fix"
SCANPATH_ANSWERS = "consolidated_answers.json"

USER_AGENT = "gazeshap-vqa/0.1 (research; +https://github.com/szzexpoi/AiR)"
CHUNK_BYTES = 1 << 20


class DownloadError(RuntimeError):
    """Raised when a download is missing, incomplete or structurally wrong."""


def make_session(retries: int = 5) -> requests.Session:
    """Build a session that retries on transient server errors.

    The Stanford mirror intermittently answers 503, and reading a remote zip
    issues hundreds of range requests, so a single failure must not abort the run.

    Args:
        retries: Number of retries per request.

    Returns:
        A configured requests session.
    """
    session = requests.Session()
    policy = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=policy, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = USER_AGENT
    return session


@dataclass
class DataPaths:
    """Canonical locations under ``data/raw``."""

    root: Path

    @property
    def raw(self) -> Path:
        """Root of the unmodified downloads."""
        return self.root / "raw"

    @property
    def images(self) -> Path:
        """Directory of Visual Genome JPGs, named ``{image_id}.jpg``."""
        return self.raw / "images"

    @property
    def air_d(self) -> Path:
        """Root of the AiR-D download."""
        return self.raw / "air_d"

    @property
    def aggregated_maps(self) -> Path:
        """Directory holding ``fixmaps``, ``fixmaps_corr``, ``fixmaps_incorr``."""
        return self.air_d / "aggregated_maps"

    @property
    def temporal_maps(self) -> Path:
        """Directory holding the ``*_1s`` / ``*_2s`` binned maps."""
        return self.air_d / "temporal_maps"

    @property
    def scanpaths(self) -> Path:
        """Directory holding ``fix/*.mat`` and ``consolidated_answers.json``."""
        return self.air_d / "scanpaths"

    @property
    def holdout(self) -> Path:
        """Optional hold-out metadata (QA pairs only; no gaze)."""
        return self.air_d / "holdout"

    @property
    def gqa(self) -> Path:
        """Directory holding ``val_balanced_questions.json``."""
        return self.raw / "gqa"

    @property
    def questions_json(self) -> Path:
        """The GQA balanced validation questions."""
        return self.gqa / GQA_QUESTIONS_MEMBER

    @property
    def manifest(self) -> Path:
        """Provenance record written after a successful download."""
        return self.raw / "manifest.json"

    def mkdirs(self) -> None:
        """Create every directory this layout refers to."""
        for p in (self.images, self.air_d, self.gqa):
            p.mkdir(parents=True, exist_ok=True)


# --- Remote zip reading -----------------------------------------------------


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests.

    Lets :mod:`zipfile` open a remote archive and read a single member without
    transferring the whole file.
    """

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        """Open the remote resource.

        Args:
            url: URL of the remote file.
            session: Optional requests session to reuse connections.

        Raises:
            DownloadError: If the server does not advertise byte-range support.
        """
        self._url = url
        self._session = session or make_session()
        head = self._session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise DownloadError(f"{url} does not support HTTP range requests")
        self._size = int(head.headers["Content-Length"])
        self._pos = 0

    def readable(self) -> bool:
        """Always ``True``."""
        return True

    def seekable(self) -> bool:
        """Always ``True``."""
        return True

    def tell(self) -> int:
        """Current read offset."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read offset.

        Args:
            offset: Byte offset.
            whence: One of ``io.SEEK_SET``, ``io.SEEK_CUR``, ``io.SEEK_END``.

        Returns:
            The new absolute offset.
        """
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        """Fill ``buffer`` from the current offset.

        Args:
            buffer: Writable buffer.

        Returns:
            Number of bytes written.
        """
        n = min(len(buffer), self._size - self._pos)
        if n <= 0:
            return 0
        end = self._pos + n - 1
        resp = self._session.get(
            self._url,
            headers={"Range": f"bytes={self._pos}-{end}", "User-Agent": USER_AGENT},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.content[:n]
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


def extract_member_over_http(url: str, member: str, dest: Path) -> Path:
    """Extract one member of a remote zip via range requests.

    Args:
        url: URL of the zip archive.
        member: Name of the member, matched on the basename if not found exactly.
        dest: Destination file path.

    Returns:
        The path written.

    Raises:
        DownloadError: If the member is not present in the archive.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = HttpRangeFile(url, session=make_session())
    with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=CHUNK_BYTES)) as zf:  # type: ignore[arg-type]
        names = zf.namelist()
        match = (
            member if member in names else next((n for n in names if Path(n).name == member), None)
        )
        if match is None:
            raise DownloadError(f"{member!r} not found in {url} (has {len(names)} members)")
        info = zf.getinfo(match)
        logger.info(
            "Extracting %s (%.1f MB uncompressed) from %s", match, info.file_size / 1e6, url
        )
        with zf.open(match) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out, CHUNK_BYTES)
    return dest


# --- Google Drive -----------------------------------------------------------


def download_drive_archive(file_id: str, dest_dir: Path, label: str) -> Path:
    """Download a Google Drive archive and unpack it.

    Args:
        file_id: Google Drive file ID.
        dest_dir: Directory the archive is unpacked into.
        label: Human-readable name used in log messages and the cache filename.

    Returns:
        The directory the archive was unpacked into.

    Raises:
        DownloadError: If the download fails or produces no files.
    """
    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)
    cache = dest_dir.parent / f".{label}.zip"
    if not cache.exists():
        logger.info("Downloading %s from Google Drive (%s)", label, file_id)
        try:
            got = gdown.download(id=file_id, output=str(cache), quiet=False, resume=True, retries=3)
        except Exception as exc:  # noqa: BLE001 - gdown raises a zoo of exceptions
            raise DownloadError(f"Google Drive download of {label} failed: {exc}") from exc
        if got is None or not cache.exists():
            raise DownloadError(
                f"Google Drive refused to serve {label} ({file_id}). This usually means the "
                f"quota is exhausted; download it manually to {cache} and re-run."
            )
    logger.info("Unpacking %s", cache.name)
    try:
        shutil.unpack_archive(str(cache), str(dest_dir))
    except shutil.ReadError as exc:
        raise DownloadError(f"{cache} is not a readable archive: {exc}") from exc
    if not any(dest_dir.iterdir()):
        raise DownloadError(f"{label} unpacked to an empty directory ({dest_dir})")
    return dest_dir


def _find_dir(root: Path, name: str) -> Path | None:
    """Locate a directory by name anywhere below ``root``.

    Args:
        root: Directory to search.
        name: Directory name to look for.

    Returns:
        The first match, or ``None``.
    """
    if not root.is_dir():
        return None
    if root.name == name:
        return root
    return next((p for p in root.rglob(name) if p.is_dir()), None)


def normalise_air_d_layout(paths: DataPaths) -> None:
    """Move the unpacked AiR-D contents into the canonical layout.

    The two Drive archives nest their content under their own top-level names
    (``AiR-D/`` for the maps, ``air-fixations/`` for the scanpaths), so the
    relevant directories are located by name and moved into
    ``aggregated_maps/``, ``temporal_maps/`` and ``scanpaths/``.

    Args:
        paths: Data paths for the current data root.

    Raises:
        DownloadError: If the aggregated maps cannot be found afterwards.
    """
    moved = 0

    # The maps archive already groups the directories the way we want them.
    for target in (paths.aggregated_maps, paths.temporal_maps):
        if target.is_dir():
            continue
        found = _find_dir(paths.air_d, target.name)
        if found is not None:
            shutil.move(str(found), str(target))
            moved += 1

    # Fallback for a flat archive: collect the map directories individually.
    paths.aggregated_maps.mkdir(parents=True, exist_ok=True)
    for name in AGGREGATED_MAP_DIRS:
        target = paths.aggregated_maps / name
        if target.is_dir():
            continue
        found = _find_dir(paths.air_d, name)
        if found is not None:
            shutil.move(str(found), str(target))
            moved += 1

    paths.scanpaths.mkdir(parents=True, exist_ok=True)
    fix = _find_dir(paths.air_d, SCANPATH_DIR)
    if fix is not None and fix.parent != paths.scanpaths:
        shutil.move(str(fix), str(paths.scanpaths / SCANPATH_DIR))
        moved += 1
    answers = next(
        (p for p in paths.air_d.rglob(SCANPATH_ANSWERS) if p.parent != paths.scanpaths), None
    )
    if answers is not None:
        shutil.move(str(answers), str(paths.scanpaths / SCANPATH_ANSWERS))
        moved += 1

    _prune_empty_dirs(paths.air_d)
    logger.info("Normalised AiR-D layout (%d directories moved)", moved)
    if not (paths.aggregated_maps / "fixmaps").is_dir():
        raise DownloadError(
            f"Expected {paths.aggregated_maps / 'fixmaps'} after unpacking the AiR-D archive. "
            f"Found instead: {sorted(p.name for p in paths.air_d.iterdir())}"
        )


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty directories left behind by the normalisation, depth first.

    Args:
        root: Directory to clean; ``root`` itself is never removed.
    """
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


# --- Images -----------------------------------------------------------------


def _fetch_image(image_id: str, dest: Path, session: requests.Session) -> tuple[str, bool, str]:
    """Fetch one Visual Genome image, trying both corpus halves.

    Args:
        image_id: Visual Genome / GQA image ID.
        dest: Destination JPG path.
        session: Requests session.

    Returns:
        Tuple of image ID, success flag and a short status string.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return image_id, True, "cached"
    for url in VG_IMAGE_URLS:
        try:
            resp = session.get(
                url.format(image_id=image_id), timeout=60, headers={"User-Agent": USER_AGENT}
            )
        except requests.RequestException as exc:
            return image_id, False, f"error: {exc}"
        if resp.status_code == 404:
            continue
        if not resp.ok:
            return image_id, False, f"http {resp.status_code}"
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return image_id, True, "downloaded"
    return image_id, False, "not found in VG_100K or VG_100K_2"


def download_images(
    image_ids: Iterable[str], dest_dir: Path, max_workers: int = 16
) -> tuple[int, list[str]]:
    """Download Visual Genome images by ID.

    Args:
        image_ids: Image IDs to fetch.
        dest_dir: Directory the JPGs are written to.
        max_workers: Number of parallel HTTP workers.

    Returns:
        Tuple of the number of images present afterwards and the list of IDs
        that could not be fetched.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(image_ids))
    failures: list[str] = []
    session = make_session()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_image, i, dest_dir / f"{i}.jpg", session): i for i in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images", unit="img"):
            image_id, ok, status = fut.result()
            if not ok:
                failures.append(image_id)
                logger.warning("Image %s failed: %s", image_id, status)
    return len(ids) - len(failures), failures


# --- Manifest ---------------------------------------------------------------


def _sha256(path: Path, limit_bytes: int | None = None) -> str:
    """Hash a file, optionally only its first ``limit_bytes``.

    Args:
        path: File to hash.
        limit_bytes: If given, hash only this many leading bytes.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    remaining = limit_bytes if limit_bytes is not None else -1
    with path.open("rb") as fh:
        while True:
            want = CHUNK_BYTES if remaining < 0 else min(CHUNK_BYTES, remaining)
            if want == 0:
                break
            chunk = fh.read(want)
            if not chunk:
                break
            h.update(chunk)
            if remaining > 0:
                remaining -= len(chunk)
    return h.hexdigest()


@dataclass
class ManifestEntry:
    """One recorded artefact in the download manifest."""

    path: str
    kind: str
    bytes: int
    sha256: str | None = None
    count: int | None = None


@dataclass
class DownloadReport:
    """Outcome of a download run."""

    entries: list[ManifestEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialise for ``manifest.json``."""
        return {
            "schema_version": 1,
            "entries": [vars(e) for e in self.entries],
            "warnings": self.warnings,
        }


def _describe(path: Path, root: Path) -> ManifestEntry:
    """Build a manifest entry for a file or directory.

    Args:
        path: File or directory to describe.
        root: Root the recorded path is made relative to.

    Returns:
        The manifest entry.
    """
    rel = str(path.relative_to(root))
    if path.is_file():
        return ManifestEntry(rel, "file", path.stat().st_size, sha256=_sha256(path))
    files = [p for p in path.rglob("*") if p.is_file()]
    return ManifestEntry(rel, "dir", sum(p.stat().st_size for p in files), count=len(files))


# --- Orchestration ----------------------------------------------------------


MAP_SUFFIXES = {".png", ".jpg", ".npy"}


def _map_qids(directory: Path) -> set[str]:
    """Collect question IDs from saliency map filenames in a directory.

    Args:
        directory: Directory of per-question map files.

    Returns:
        Set of question IDs, empty if the directory does not exist.
    """
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.iterdir() if p.suffix.lower() in MAP_SUFFIXES}


def load_air_d_qids(paths: DataPaths) -> list[str]:
    """Return the canonical AiR-D question IDs.

    ``fixmaps`` ships 1,454 maps, but only 1,422 of those questions also have
    temporal bins, scanpaths and recorded answers. Since Stage 1 supervision
    needs the 1-2s and 2-3s bins (idea.md 4.3), the canonical set is the
    intersection; the surplus aggregated maps are ignored.

    Args:
        paths: Data paths for the current data root.

    Returns:
        Sorted list of question IDs.

    Raises:
        DownloadError: If the ``fixmaps`` directory is missing or empty.
    """
    fixmaps = paths.aggregated_maps / "fixmaps"
    if not fixmaps.is_dir():
        raise DownloadError(f"Missing {fixmaps}; run the AiR-D download first.")
    qids = _map_qids(fixmaps)
    if not qids:
        raise DownloadError(f"{fixmaps} contains no saliency maps.")

    for bin_name in ("1s", "2s"):
        binned = _map_qids(paths.temporal_maps / f"fixmaps_{bin_name}")
        if binned:
            qids &= binned
    answers = paths.scanpaths / SCANPATH_ANSWERS
    if answers.is_file():
        recorded = json.loads(answers.read_text(encoding="utf-8")).get("question", {})
        if recorded:
            qids &= set(recorded)
    return sorted(qids)


def image_ids_for_qids(
    questions: dict[str, dict], qids: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Map AiR-D question IDs onto GQA image IDs.

    Args:
        questions: The parsed GQA question JSON, keyed by question ID.
        qids: AiR-D question IDs.

    Returns:
        Tuple of the image IDs and the question IDs missing from the GQA file.
    """
    image_ids: list[str] = []
    missing: list[str] = []
    for qid in qids:
        entry = questions.get(qid)
        if entry is None:
            missing.append(qid)
            continue
        image_ids.append(str(entry["imageId"]))
    return sorted(set(image_ids)), missing


def download_all(
    data_root: Path,
    *,
    skip_images: bool = False,
    include_holdout: bool = True,
    max_workers: int = 16,
) -> DownloadReport:
    """Fetch every source dataset into ``data_root`` and write the manifest.

    Args:
        data_root: The ``data/`` directory.
        skip_images: Skip the Visual Genome image download.
        include_holdout: Attempt the optional hold-out metadata download.
        max_workers: Parallel workers for the image download.

    Returns:
        The download report, also written to ``data/raw/manifest.json``.

    Raises:
        DownloadError: If a required source is missing or structurally wrong.
    """
    paths = DataPaths(Path(data_root))
    paths.mkdirs()
    report = DownloadReport()

    # 1. AiR-D saliency maps and scanpaths.
    download_drive_archive(AIR_D_MAPS_DRIVE_ID, paths.air_d, "air_d_maps")
    download_drive_archive(AIR_D_SCANPATHS_DRIVE_ID, paths.air_d, "air_d_scanpaths")
    normalise_air_d_layout(paths)

    qids = load_air_d_qids(paths)
    logger.info("AiR-D provides %d questions", len(qids))
    if len(qids) != AIR_D_EXPECTED_QUESTIONS:
        report.warnings.append(
            f"expected {AIR_D_EXPECTED_QUESTIONS} AiR-D questions, found {len(qids)}"
        )

    # 2. GQA questions: one member out of a 1.5 GB zip.
    if not paths.questions_json.exists():
        extract_member_over_http(GQA_QUESTIONS_ZIP_URL, GQA_QUESTIONS_MEMBER, paths.questions_json)
    questions = json.loads(paths.questions_json.read_text(encoding="utf-8"))
    logger.info("GQA balanced val holds %d questions", len(questions))

    image_ids, missing_qids = image_ids_for_qids(questions, qids)
    if missing_qids:
        report.warnings.append(
            f"{len(missing_qids)} AiR-D qids absent from {GQA_QUESTIONS_MEMBER} "
            f"(first few: {missing_qids[:5]})"
        )
    logger.info("AiR-D covers %d distinct images", len(image_ids))
    if image_ids and abs(len(image_ids) - AIR_D_EXPECTED_IMAGES) > 5:
        report.warnings.append(
            f"expected ~{AIR_D_EXPECTED_IMAGES} images, resolved {len(image_ids)}"
        )

    # 3. Images, one by one from Visual Genome.
    if not skip_images:
        n_ok, failures = download_images(image_ids, paths.images, max_workers=max_workers)
        if failures:
            raise DownloadError(
                f"{len(failures)} of {len(image_ids)} images could not be fetched "
                f"(first few: {failures[:5]}). Refusing to leave a partial dataset."
            )
        logger.info("Images present: %d", n_ok)

    # 4. Optional hold-out metadata (QA pairs only; the gaze stays private).
    if include_holdout:
        for name, file_id in AIR_D_HOLDOUT_DRIVE_IDS.items():
            try:
                download_drive_archive(file_id, paths.holdout / name, f"holdout_{name}")
            except DownloadError as exc:
                report.warnings.append(f"optional hold-out {name} unavailable: {exc}")
                logger.warning("Optional hold-out %s unavailable: %s", name, exc)

    # 5. Manifest.
    for target in (
        paths.questions_json,
        paths.aggregated_maps,
        paths.temporal_maps,
        paths.scanpaths,
        paths.images,
    ):
        if target.exists():
            report.entries.append(_describe(target, paths.raw))
    paths.manifest.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", paths.manifest)
    return report


def verify(data_root: Path, *, require_images: bool = True) -> list[str]:
    """Check that the downloaded data is complete and self-consistent.

    Args:
        data_root: The ``data/`` directory.
        require_images: Whether missing images count as a problem.

    Returns:
        A list of human-readable problems; empty means everything is in place.
    """
    paths = DataPaths(Path(data_root))
    problems: list[str] = []

    if not (paths.aggregated_maps / "fixmaps").is_dir():
        problems.append(f"missing {paths.aggregated_maps / 'fixmaps'}")
        return problems
    if not (paths.scanpaths / SCANPATH_DIR).is_dir():
        problems.append(f"missing {paths.scanpaths / SCANPATH_DIR}")
    if not paths.questions_json.exists():
        problems.append(f"missing {paths.questions_json}")
        return problems

    qids = load_air_d_qids(paths)
    questions = json.loads(paths.questions_json.read_text(encoding="utf-8"))
    image_ids, missing_qids = image_ids_for_qids(questions, qids)
    if missing_qids:
        problems.append(f"{len(missing_qids)} qids not in the GQA question file")
    if require_images:
        absent = [i for i in image_ids if not (paths.images / f"{i}.jpg").exists()]
        if absent:
            problems.append(f"{len(absent)} images missing from {paths.images}")
    return problems
