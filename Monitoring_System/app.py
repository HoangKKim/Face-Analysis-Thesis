import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import tempfile
import threading
from zipfile import ZipFile
import cv2

from src.main import main 

def zip_folder(folder_path, zip_path):
    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")
        self.root.geometry("600x600")
        self.root.resizable(False, False)

        self.video_path = None
        self.zip_path = None
        self.temp_dir = tempfile.mkdtemp()
        self.preview_img = None

        self.label = tk.Label(root, text="No video selected")
        self.label.pack(pady=10)

        self.preview_label = tk.Label(root)
        self.preview_label.pack()

        self.browse_button = tk.Button(root, text="Select Video", command=self.browse_video)
        self.browse_button.pack(pady=5)

        self.process_button = tk.Button(root, text="Process Video", command=self.process_video, state=tk.DISABLED)
        self.process_button.pack(pady=5)

        self.download_button = tk.Button(root, text="Download ZIP", command=self.download_zip, state=tk.DISABLED)
        self.download_button.pack(pady=5)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=400, mode='indeterminate')
        self.progress.pack(pady=10)

        # ===== Add log display widget =====
        self.log_text = tk.Text(root, height=10, wrap="word", state=tk.DISABLED)
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=False)

    def log_message(self, message):
        def append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, append)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if path:
            self.video_path = path
            self.label.config(text=f"Selected: {os.path.basename(path)}")
            self.process_button.config(state=tk.NORMAL)
            self.preview_first_frame(path)

    def preview_first_frame(self, path):
        cap = cv2.VideoCapture(path)
        success, frame = cap.read()
        cap.release()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((320, 240))
            self.preview_img = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_img)
        else:
            self.preview_label.config(image='', text="Could not load preview")

    def process_video(self):
        thread = threading.Thread(target=self._run_processing)
        thread.start()

    def _run_processing(self):
        self.progress.start()
        self.log_message("🟡 Processing started...")
        try:
            output_folder = main(self.video_path, log_callback=self.log_message)
            self.zip_path = os.path.join(self.temp_dir, "output.zip")
            zip_folder(output_folder, self.zip_path)

            self.download_button.config(state=tk.NORMAL)
            self.log_message("✅ Zipping completed.")
            messagebox.showinfo("Success", "Video processed and zipped successfully.")
        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()
            self.log_message("🔚 Done.")

    def download_zip(self):
        if self.zip_path and os.path.exists(self.zip_path):
            save_path = filedialog.asksaveasfilename(defaultextension=".zip",
                                                     filetypes=[("ZIP files", "*.zip")])
            if save_path:
                with open(self.zip_path, "rb") as src, open(save_path, "wb") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Saved", "File downloaded successfully.")
        else:
            messagebox.showwarning("Warning", "No zip file available.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
