import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps


def _gather_images(root_folder: str) -> list[str]:
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = []
    for root_dir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(supported_exts):
                paths.append(os.path.join(root_dir, file))
    return paths


def invert_images(image_paths: list[str], output_dir: str, progress: ttk.Progressbar | None = None) -> None:
    for i, path in enumerate(image_paths, 1):
        img = Image.open(path)
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb_image = Image.merge("RGB", (r, g, b))
            inverted_rgb = ImageOps.invert(rgb_image)
            inverted = Image.merge("RGBA", (*inverted_rgb.split(), a))
        else:
            inverted = ImageOps.invert(img.convert("RGB"))
        name, ext = os.path.splitext(os.path.basename(path))
        folder_name = os.path.basename(os.path.dirname(path))
        new_name = f"{name}_{folder_name}{ext}"
        inverted.save(os.path.join(output_dir, new_name))
        if progress is not None:
            progress["value"] = i / len(image_paths) * 100
            progress.update()

def main():
    root = tk.Tk()
    root.title("Invert Images")

    root_folder = filedialog.askdirectory(title="Quellordner auswählen (inkl. Unterordner)")
    if not root_folder:
        messagebox.showwarning("Abbruch", "Kein Quellordner ausgewählt.")
        return

    output_dir = filedialog.askdirectory(title="Zielordner zum Speichern auswählen")
    if not output_dir:
        messagebox.showwarning("Abbruch", "Kein Zielordner ausgewählt.")
        return

    images = _gather_images(root_folder)
    if not images:
        messagebox.showinfo("Keine Bilder", "Keine unterstützten Dateien gefunden.")
        return

    progress = ttk.Progressbar(root, length=300)
    progress.pack(pady=10)

    def worker():
        invert_images(images, output_dir, progress)
        messagebox.showinfo("Fertig", f"Invertierte Bilder wurden in:\n{output_dir}\ngespeichert.")
        root.quit()

    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    main()
