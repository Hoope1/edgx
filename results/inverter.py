import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageOps

def invert_images_in_folder(root_folder, output_dir):
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for root_dir, _, files in os.walk(root_folder):
        folder_name = os.path.basename(root_dir)
        for file in files:
            if file.lower().endswith(supported_exts):
                input_path = os.path.join(root_dir, file)
                img = Image.open(input_path)
                # Handle transparency by preserving alpha channel
                if img.mode == 'RGBA':
                    r, g, b, a = img.split()
                    rgb_image = Image.merge("RGB", (r, g, b))
                    inverted_rgb = ImageOps.invert(rgb_image)
                    inverted = Image.merge("RGBA", (*inverted_rgb.split(), a))
                else:
                    inverted = ImageOps.invert(img.convert('RGB'))
                name, ext = os.path.splitext(file)
                new_name = f"{name}_{folder_name}{ext}"
                output_path = os.path.join(output_dir, new_name)
                inverted.save(output_path)

def main():
    root = tk.Tk()
    root.withdraw()

    # Select a single root folder
    root_folder = filedialog.askdirectory(title="Quellordner auswählen (inkl. Unterordner)")
    if not root_folder:
        messagebox.showwarning("Abbruch", "Kein Quellordner ausgewählt.")
        return

    output_dir = filedialog.askdirectory(title="Zielordner zum Speichern auswählen")
    if not output_dir:
        messagebox.showwarning("Abbruch", "Kein Zielordner ausgewählt.")
        return

    invert_images_in_folder(root_folder, output_dir)
    messagebox.showinfo("Fertig", f"Invertierte Bilder wurden in:\n{output_dir}\ngespeichert.")

if __name__ == "__main__":
    main()
