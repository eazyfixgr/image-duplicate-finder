"""Image Duplicate Finder.

Finds the same picture saved with different names / sizes / compression
(Viber, Messenger, etc.) using a perceptual hash. It never deletes anything:
every duplicate is moved to a ``duplicates`` folder inside the chosen folder.
"""

import argparse
import os
import queue
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND_AVAILABLE = True
except ImportError:  # drag & drop is optional; the app still works without it
    DND_FILES = None
    TkinterDnD = None
    TKDND_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - shown when deps are missing
    try:
        _r = tk.Tk()
        _r.withdraw()
        messagebox.showerror(
            "Image Duplicate Finder",
            f"Λείπουν βιβλιοθήκες: {exc}\n\nΤρέξε:  pip install pillow numpy")
        _r.destroy()
    except Exception:
        pass
    raise SystemExit(1)


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".avif",
}

HASH_SIZE = 8
DCT_SIZE = 32


def _dct_matrix(n):
    k = np.arange(n, dtype=np.float64)[:, None]
    x = np.arange(n, dtype=np.float64)[None, :]
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] = np.sqrt(1.0 / n)
    return m


_DCT = _dct_matrix(DCT_SIZE)


@dataclass
class ImageInfo:
    path: Path
    phash: int
    dhash: int
    aspect: float
    pixels: int
    size: int


def _bits_to_int(bits):
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def analyze_image(path):
    """Return the perceptual hashes and basic info of one image."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        width, height = im.size
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        gray = im.convert("L")
        arr32 = np.asarray(gray.resize((DCT_SIZE, DCT_SIZE), Image.Resampling.LANCZOS),
                           dtype=np.float64)
        arr98 = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)

    dct = _DCT @ arr32 @ _DCT.T
    low = dct[:HASH_SIZE, :HASH_SIZE]
    phash = _bits_to_int((low > np.median(low)).flatten())
    dhash = _bits_to_int((arr98[:, 1:] > arr98[:, :-1]).flatten())
    aspect = (width / height) if height else 0.0
    return ImageInfo(path=Path(path), phash=phash, dhash=dhash, aspect=aspect,
                     pixels=width * height, size=Path(path).stat().st_size)


def _same_picture(a, b, threshold):
    """True when two images look like the same picture.

    The aspect ratio guard keeps different photos with a similar hash apart.
    """
    if a.aspect > 0 and b.aspect > 0:
        if abs(a.aspect - b.aspect) / max(a.aspect, b.aspect) > 0.08:
            return False
    if (a.phash ^ b.phash).bit_count() <= threshold:
        return True
    return (a.dhash ^ b.dhash).bit_count() <= 5


def find_groups(infos, threshold, cancelled=None, progress=None):
    """Group images that are duplicates of each other (union-find)."""
    parent = list(range(len(infos)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri

    n = len(infos)
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        if cancelled is not None and cancelled.is_set():
            return []
        a = infos[i]
        for j in range(i + 1, n):
            if _same_picture(a, infos[j], threshold):
                union(i, j)
        done += n - i - 1
        if progress is not None and total:
            progress(done, total)

    buckets = {}
    for i, info in enumerate(infos):
        buckets.setdefault(root(i), []).append(info)
    groups = [g for g in buckets.values() if len(g) > 1]
    groups.sort(key=len, reverse=True)
    return groups


def choose_keeper(group):
    """Keep the biggest / highest quality copy of a duplicate group."""
    return max(group, key=lambda info: (info.pixels, info.size))


def collect_images(folder, skip=None):
    folder = Path(folder)
    found = []
    for dirpath, dirnames, filenames in os.walk(folder):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if skip is None or
                       (current / d).resolve() != skip.resolve()]
        for name in filenames:
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(current / name)
    return sorted(found, key=lambda p: p.name.lower())


def unique_destination(folder, name):
    dest = folder / name
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    number = 2
    while True:
        candidate = folder / f"{stem} ({number}){suffix}"
        if not candidate.exists():
            return candidate
        number += 1


def move_duplicates(groups, duplicates_dir, log=print):
    """Move every non-keeper into ``duplicates_dir``. Returns (moved, errors)."""
    duplicates_dir = Path(duplicates_dir)
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    moved, errors = 0, []
    for group in groups:
        keeper = choose_keeper(group)
        for info in group:
            if info.path == keeper.path:
                continue
            try:
                dest = unique_destination(duplicates_dir, info.path.name)
                shutil.move(str(info.path), str(dest))
                log(f"Μετακίνηση: {info.path}  ->  {dest}")
                moved += 1
            except Exception as exc:
                errors.append((info.path, str(exc)))
                log(f"Σφάλμα: {info.path}: {exc}")
    return moved, errors


# --------------------------------------------------------------------- GUI
class DuplicateFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Duplicate Finder")
        self.root.geometry("980x700")
        self.root.minsize(760, 540)

        self._events = queue.Queue()
        self._cancel = threading.Event()
        self._thread = None
        self._groups = []
        self._folder = None

        self.folder_var = tk.StringVar()
        self.threshold_var = tk.IntVar(value=8)
        self.threshold_label = tk.StringVar(value="8 / 64")
        self.status_var = tk.StringVar(value="Διάλεξε φάκελο για αρχή.")

        self._build()
        self._enable_drag_and_drop()
        self.root.after(80, self._poll_events)

    def _enable_drag_and_drop(self):
        if not TKDND_AVAILABLE:
            return
        try:
            for widget in (self.root, self.folder_entry):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = (event.data,)
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                chosen = path
            elif path.is_file():
                chosen = path.parent
            else:
                continue
            self.folder_var.set(str(chosen))
            self.status_var.set(f"Φάκελος: {chosen} — πάτα «Σάρωση».")
            return "break"
        messagebox.showwarning("Image Duplicate Finder",
                               "Σύρε εδώ έναν φάκελο με εικόνες.")
        return "break"

    def _build(self):
        body = ttk.Frame(self.root, padding=12)
        body.pack(fill='both', expand=True)

        top = ttk.Frame(body)
        top.pack(fill='x')
        ttk.Label(top, text="Φάκελος με εικόνες:").pack(side='left')
        self.folder_entry = ttk.Entry(top, textvariable=self.folder_var)
        self.folder_entry.pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(top, text="Επιλογή...", command=self.pick_folder).pack(side='left')

        hint = ("Σύρε κι άφησε εδώ έναν φάκελο (drag & drop)"
                if TKDND_AVAILABLE else
                "Drag & drop μη διαθέσιμο — διάλεξε φάκελο με το κουμπί")
        ttk.Label(body, text=hint, foreground="#666").pack(fill='x', pady=(4, 0))

        opts = ttk.Frame(body)
        opts.pack(fill='x', pady=(10, 0))
        ttk.Label(opts, text="Ανοχή ομοιότητας:").pack(side='left')
        ttk.Scale(opts, from_=0, to=20, orient='horizontal', length=260,
                  variable=self.threshold_var,
                  command=lambda _v: self._threshold_changed()).pack(side='left', padx=8)
        ttk.Label(opts, textvariable=self.threshold_label, width=8).pack(side='left')
        ttk.Label(opts, foreground="#666",
                  text="(μικρότερο = πιο αυστηρό)").pack(side='left', padx=(4, 0))

        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(10, 0))
        self.scan_btn = ttk.Button(buttons, text="Σάρωση για διπλότυπα",
                                   command=self.start_scan)
        self.scan_btn.pack(side='left')
        self.cancel_btn = ttk.Button(buttons, text="Ακύρωση", command=self.cancel,
                                     state='disabled')
        self.cancel_btn.pack(side='left', padx=6)
        self.move_btn = ttk.Button(buttons, text="Μετακίνηση στο duplicates",
                                   command=self.move_found, state='disabled')
        self.move_btn.pack(side='left', padx=6)

        self.progress = ttk.Progressbar(body, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=(12, 4))
        ttk.Label(body, textvariable=self.status_var).pack(fill='x')

        self.results = scrolledtext.ScrolledText(
            body, height=20, wrap='word', state='disabled')
        self.results.pack(fill='both', expand=True, pady=(8, 0))

        ttk.Label(body, foreground="#666",
                  text="Καμία εικόνα δεν διαγράφεται. Τα διπλότυπα μετακινούνται "
                       "στον υποφάκελο «duplicates» του φακέλου που διάλεξες."
                  ).pack(fill='x', pady=(6, 0))

    # ------------------------------------------------------------- helpers
    def _threshold_changed(self):
        self.threshold_label.set(f"{self.threshold_var.get()} / 64")

    def _log(self, text):
        self.results.configure(state='normal')
        self.results.insert('end', text + "\n")
        self.results.see('end')
        self.results.configure(state='disabled')

    def _clear_log(self):
        self.results.configure(state='normal')
        self.results.delete('1.0', 'end')
        self.results.configure(state='disabled')

    def _set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        self.scan_btn.configure(state=state)
        self.folder_entry.configure(state=state)
        self.cancel_btn.configure(state='normal' if busy else 'disabled')
        if busy:
            self.move_btn.configure(state='disabled')

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Διάλεξε φάκελο με εικόνες")
        if folder:
            self.folder_var.set(folder)

    def cancel(self):
        if self._thread is not None and self._thread.is_alive():
            self._cancel.set()
            self.status_var.set("Ακύρωση...")

    # ---------------------------------------------------------- event pump
    def _post(self, kind, payload=None):
        self._events.put((kind, payload))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self._events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _handle_event(self, kind, payload):
        if kind == "status":
            self.status_var.set(payload)
        elif kind == "progress":
            done, total = payload
            self.progress.configure(maximum=max(1, total), value=done)
        elif kind == "log":
            self._log(payload)
        elif kind == "done":
            self._scan_done(payload)
        elif kind == "moved":
            self._move_done(payload)
        elif kind == "error":
            self._set_busy(False)
            self.status_var.set("Σφάλμα.")
            messagebox.showerror("Image Duplicate Finder", payload)
        elif kind == "cancelled":
            self._set_busy(False)
            self.status_var.set("Ακυρώθηκε.")

    # -------------------------------------------------------------- scan
    def start_scan(self):
        folder = Path(self.folder_var.get().strip() or ".")
        if not folder.is_dir():
            messagebox.showwarning(
                "Image Duplicate Finder", "Διάλεξε πρώτα έναν φάκελο που υπάρχει.")
            return
        self._folder = folder
        self._groups = []
        self._cancel.clear()
        self._clear_log()
        self.progress.configure(value=0, maximum=100)
        self._set_busy(True)
        self.status_var.set("Σάρωση...")
        threshold = int(self.threshold_var.get())
        self._thread = threading.Thread(target=self._scan_worker,
                                        args=(folder, threshold), daemon=True)
        self._thread.start()

    def _scan_worker(self, folder, threshold):
        try:
            duplicates_dir = folder / "duplicates"
            files = collect_images(folder, skip=duplicates_dir)
            self._post("log", f"Βρέθηκαν {len(files)} εικόνες στον φάκελο.")
            if not files:
                self._post("done", {"total": 0, "groups": [], "skipped": []})
                return

            infos, skipped = [], []
            for index, path in enumerate(files, start=1):
                if self._cancel.is_set():
                    self._post("cancelled")
                    return
                try:
                    infos.append(analyze_image(path))
                except Exception as exc:
                    skipped.append((path, str(exc)))
                if index % 5 == 0 or index == len(files):
                    self._post("progress", (index, len(files)))
                    self._post("status", f"Ανάλυση εικόνων... {index}/{len(files)}")

            self._post("status", "Σύγκριση εικόνων...")
            self._post("progress", (0, 100))

            def compare_progress(done, total):
                self._post("progress", (done, total))
                if done % max(1, total // 20) < max(1, total // 20):
                    percent = int(done * 100 / total) if total else 100
                    self._post("status", f"Σύγκριση εικόνων... {percent}%")

            groups = find_groups(infos, threshold, cancelled=self._cancel,
                                 progress=compare_progress)
            if self._cancel.is_set():
                self._post("cancelled")
                return
            self._post("done", {"total": len(files), "groups": groups,
                                "skipped": skipped})
        except Exception as exc:
            self._post("error", str(exc))

    def _scan_done(self, payload):
        self._set_busy(False)
        self.progress.configure(value=100)
        groups = payload["groups"]
        self._groups = groups
        duplicates = sum(len(g) - 1 for g in groups)

        self._log("")
        if not groups:
            self._log("Δεν βρέθηκαν διπλότυπες εικόνες.")
        for number, group in enumerate(groups, start=1):
            keeper = choose_keeper(group)
            self._log(f"ΟΜΑΔΑ {number} — {len(group)} εικόνες")
            self._log(f"  Κρατάω: {keeper.path}  "
                      f"({keeper.pixels} px, {keeper.size / 1024:.0f} KB)")
            for info in group:
                if info.path == keeper.path:
                    continue
                distance = (info.phash ^ keeper.phash).bit_count()
                self._log(f"  Διπλότυπο: {info.path}  (διαφορά {distance}/64)")
            self._log("")

        for path, reason in payload.get("skipped", []):
            self._log(f"Παράλειψη (δεν διαβάστηκε): {path} — {reason}")

        if groups:
            self.progress.configure(value=100, maximum=100)
            self.status_var.set(
                f"Έτοιμο: {len(groups)} ομάδες, {duplicates} διπλότυπα από "
                f"{payload['total']} εικόνες. Πάτα «Μετακίνηση».")
            self.move_btn.configure(state='normal')
        else:
            self.progress.configure(value=0)
            self.status_var.set(f"Καμία διπλότυπη εικόνα από {payload['total']}.")

    # -------------------------------------------------------------- move
    def move_found(self):
        if not self._groups or self._folder is None:
            return
        duplicates = sum(len(g) - 1 for g in self._groups)
        if not messagebox.askyesno(
                "Μετακίνηση διπλότυπων",
                f"Θα μετακινηθούν {duplicates} εικόνες στον φάκελο:\n"
                f"{self._folder / 'duplicates'}\n\n"
                "Καμία εικόνα δεν διαγράφεται. Συνέχεια;"):
            return
        self._set_busy(True)
        self.cancel_btn.configure(state='disabled')
        self.move_btn.configure(state='disabled')
        self.status_var.set("Μετακίνηση...")
        self._thread = threading.Thread(target=self._move_worker,
                                        args=(self._groups, self._folder / "duplicates"),
                                        daemon=True)
        self._thread.start()

    def _move_worker(self, groups, destination):
        try:
            moved, errors = move_duplicates(groups, destination, log=self._post_log)
            self._post("moved", {"moved": moved, "errors": errors})
        except Exception as exc:
            self._post("error", str(exc))

    def _post_log(self, text):
        self._post("log", text)

    def _move_done(self, payload):
        self._set_busy(False)
        self._groups = []
        self.move_btn.configure(state='disabled')
        self.status_var.set(f"Μετακινήθηκαν {payload['moved']} διπλότυπα. "
                            "Μπορείς να ξανακάνεις σάρωση.")
        self.progress.configure(value=0)
        if payload["errors"]:
            self._log("")
            self._log(f"Ολοκληρώθηκε με {len(payload['errors'])} σφάλματα:")
            for path, reason in payload["errors"]:
                self._log(f"  {path}: {reason}")
        messagebox.showinfo(
            "Image Duplicate Finder",
            f"Μετακινήθηκαν {payload['moved']} εικόνες στον φάκελο duplicates.\n"
            "Καμία εικόνα δεν διαγράφηκε.")


# ------------------------------------------------------------------- CLI
def cli_scan(folder, threshold, move=False):
    folder = Path(folder)
    files = collect_images(folder, skip=folder / "duplicates")
    print(f"Εικόνες: {len(files)}")
    infos, skipped = [], 0
    for path in files:
        try:
            infos.append(analyze_image(path))
        except Exception:
            skipped += 1
    groups = find_groups(infos, threshold)
    duplicates = sum(len(g) - 1 for g in groups)
    print(f"Ομάδες: {len(groups)}  Διπλότυπα: {duplicates}  "
          f"Παραλείφθηκαν: {skipped}")
    for number, group in enumerate(groups, start=1):
        keeper = choose_keeper(group)
        print(f"\nΟΜΑΔΑ {number}: κρατάω {keeper.path}")
        for info in group:
            if info.path != keeper.path:
                print(f"  -> {info.path}")
    if move and groups:
        moved, errors = move_duplicates(groups, folder / "duplicates")
        print(f"\nΜετακινήθηκαν: {moved}  Σφάλματα: {len(errors)}")


def main():
    parser = argparse.ArgumentParser(description="Image Duplicate Finder")
    parser.add_argument("--scan", metavar="FOLDER",
                        help="scan a folder without opening the GUI")
    parser.add_argument("--threshold", type=int, default=8,
                        help="pHash tolerance 0-20 (default 8)")
    parser.add_argument("--move", action="store_true",
                        help="with --scan: move duplicates into <FOLDER>/duplicates")
    args = parser.parse_args()

    if args.scan:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        cli_scan(args.scan, max(0, min(20, args.threshold)), move=args.move)
        return

    root = TkinterDnD.Tk() if TKDND_AVAILABLE else tk.Tk()
    DuplicateFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
