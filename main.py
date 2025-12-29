'''
Author: baodinhgia
Email: baodg@falcongames.com
Company: Falcon Games
Date: 2025-12-26
'''

import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from skimage import color


# DEFAULT COLOR PALETTE
DEFAULT_PALETTE_HEX = [
    "#ff90a7", "#ee444a", "#ffe201", "#ff8100", "#8f56fe",
    "#f443e0", "#00da00", "#00b587", "#148eff", "#00ecfe",
    "#e0e1eb", "#fed495", "#39414b", "#966e46", "#77422a",
    "#441b6d", "#718d24", "#403dc9", "#849ca8", "#efad29"
]

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])

class PixelArtTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Pic2Pix v1.1")
        self.root.geometry("1200x750")

        self.original_image = None
        self.processed_image = None
        
        # Palette
        self.current_palette_hex = list(DEFAULT_PALETTE_HEX)
        self.palette_rgb = [hex_to_rgb(c) for c in self.current_palette_hex]
        self.palette_np = np.array(self.palette_rgb)

        # Zoom manager
        self.zoom_scale_original = 1.0
        self.zoom_scale_processed = 1.0
        self.is_showing_original = False

        # INTERFACE
        self.control_frame = tk.Frame(root, width=350, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Select Image
        tk.Button(self.control_frame, text="Select Image", command=self.load_image, bg="#ddd").pack(fill=tk.X)

        # Pixel Size
        tk.Label(self.control_frame, text="Pixel Size (WxH):").pack(anchor="w")
        self.size_frame = tk.Frame(self.control_frame)
        self.size_frame.pack(fill=tk.X)
        self.width_var = tk.IntVar(value=64)
        self.height_var = tk.IntVar(value=64)
        tk.Entry(self.size_frame, textvariable=self.width_var, width=8).pack(side=tk.LEFT)
        tk.Label(self.size_frame, text="x").pack(side=tk.LEFT, padx=5)
        tk.Entry(self.size_frame, textvariable=self.height_var, width=8).pack(side=tk.LEFT)

        # Target Colors Count
        tk.Label(self.control_frame, text="Target Color Count:", fg="black").pack(anchor="w")
        self.target_colors_var = tk.IntVar(value=len(DEFAULT_PALETTE_HEX))
        self.color_slider = tk.Scale(self.control_frame, from_=2, to=len(DEFAULT_PALETTE_HEX), orient=tk.HORIZONTAL, variable=self.target_colors_var)
        self.color_slider.pack(fill=tk.X)

        # Custom Palette
        tk.Label(self.control_frame, text="Custom Palette (Click to change):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.palette_frame = tk.Frame(self.control_frame)
        self.palette_frame.pack(fill=tk.X)
        self.palette_buttons = []
        # 5x3 color grid
        for i in range(20):
            btn = tk.Button(self.palette_frame, bg=self.current_palette_hex[i], width=4, height=1, 
                            command=lambda idx=i: self.change_palette_color(idx))
            btn.grid(row=i//5, column=i%5, padx=2, pady=2)
            self.palette_buttons.append(btn)


        # Process
        tk.Button(self.control_frame, text="PROCESS", command=self.process_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=20)


        # Zoom & Compare
        tk.Label(self.control_frame, text="View Controls:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.view_frame = tk.Frame(self.control_frame)
        self.view_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(self.view_frame, text="Zoom In (+)", command=lambda: self.change_zoom(0.2)).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Button(self.view_frame, text="Zoom Out (-)", command=lambda: self.change_zoom(-0.2)).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.compare_btn = tk.Button(self.control_frame, text="HOLD to Compare Original", bg="#FF9800", fg="white")
        self.compare_btn.pack(fill=tk.X, pady=5)
        self.compare_btn.bind("<ButtonPress-1>", self.on_compare_press)
        self.compare_btn.bind("<ButtonRelease-1>", self.on_compare_release)


        # Stats
        tk.Label(self.control_frame, text="Stats:", font=("Arial", 10, "bold")).pack(anchor="w")
        # canvas + scrollbar container
        self.stats_container = tk.Frame(self.control_frame, bd=1, relief=tk.SUNKEN)
        self.stats_container.pack(fill=tk.BOTH, expand=True, pady=5)
        self.stats_canvas = tk.Canvas(self.stats_container, height=200) # height tùy chỉnh
        self.stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.stats_scrollbar = tk.Scrollbar(self.stats_container, orient="vertical", command=self.stats_canvas.yview)
        self.stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_canvas.configure(yscrollcommand=self.stats_scrollbar.set)
        # color frame
        self.stats_content = tk.Frame(self.stats_canvas)
        self.stats_canvas.create_window((0, 0), window=self.stats_content, anchor="nw")
        self.stats_content.bind("<Configure>", lambda e: self.stats_canvas.configure(scrollregion=self.stats_canvas.bbox("all")))

        # Export
        tk.Button(self.control_frame, text="Export (PNG)", command=self.save_image, bg="#2196F3", fg="white").pack(fill=tk.X, pady=10)

        # Right Panel - Display
        self.display_frame = tk.Frame(root, bg="#333")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.display_frame, bg="#333", fg='white', text="No image yet.")
        self.image_label.pack(expand=True)


    # --- CUSTOM PALETTE ---
    def change_palette_color(self, index):
        # Mở hộp thoại chọn màu
        color = colorchooser.askcolor(color=self.current_palette_hex[index], title="Choose Color")[1]
        if color:
            # Cập nhật dữ liệu
            self.current_palette_hex[index] = color
            self.palette_rgb[index] = hex_to_rgb(color)
            self.palette_np = np.array(self.palette_rgb) # Cập nhật lại mảng numpy
            # Cập nhật giao diện nút
            self.palette_buttons[index].config(bg=color)

    # --- ZOOM & COMPARE ---
    def change_zoom(self, factor):
        if not self.original_image: return
        
        if self.is_showing_original or self.processed_image is None:
            self.zoom_scale_original = max(0.2, min(10.0, self.zoom_scale_original + factor))
            self.show_preview(self.original_image, self.zoom_scale_original)
        else:
            self.zoom_scale_processed = max(0.2, min(10.0, self.zoom_scale_processed + factor))
            self.show_preview(self.processed_image, self.zoom_scale_processed)

    def on_compare_press(self, event):
        if self.original_image:
            self.is_showing_original = True
            self.show_preview(self.original_image, self.zoom_scale_original)

    def on_compare_release(self, event):
        self.is_showing_original = False
        if self.processed_image:
            self.show_preview(self.processed_image, self.zoom_scale_processed)
        elif self.original_image:
            self.show_preview(self.original_image, self.zoom_scale_original)

    # --- IMAGE LOAD ---
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            w, h = self.original_image.size
            ratio = w / h
            self.width_var.set(64)
            self.height_var.set(int(64 / ratio))
            
            self.zoom_scale_original = 1.0 
            self.zoom_scale_processed = 1.0 
            self.processed_image = None
            
            self.show_preview(self.original_image, self.zoom_scale_original)


    # --- IMAGE PREVIEW ---
    def show_preview(self, img, zoom_factor):
        if img is None: return

        display_w, display_h = 700, 600
        img_aspect = img.width / img.height

        if img.width > display_w or img.height > display_h:
            if display_w / display_h > img_aspect:
                base_h = display_h
                base_w = int(base_h * img_aspect)
            else:
                base_w = display_w
                base_h = int(base_w / img_aspect)
        else:
            base_w = img.width * 5
            base_h = img.height * 5

        final_w = int(base_w * zoom_factor)
        final_h = int(base_h * zoom_factor)

        img_preview = img.resize((final_w, final_h), Image.NEAREST)
        
        photo = ImageTk.PhotoImage(img_preview)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo


    # ===================== CORE: IMAGE PROCESSING ====================
    # Support: Get HSL distance
    # def get_weighted_distance(self, c1, c2):

    #     # convert RGB to HLS
    #     r1, g1, b1 = c1[0]/255.0, c1[1]/255.0, c1[2]/255.0
    #     r2, g2, b2 = c2[0]/255.0, c2[1]/255.0, c2[2]/255.0
    #     h1, l1, s1 = colorsys.rgb_to_hls(r1, g1, b1)
    #     h2, l2, s2 = colorsys.rgb_to_hls(r2, g2, b2)

    #     # grayscale logic
    #     # if saturation(s) < 0.15, hue(h) would have least to none impact on human eye
    #     # if s1 < 0.15 or s2 < 0.15:
    #     #     # get dist of light(l) and saturation now
    #     #     return abs(l1 - l2) * 2.0 + abs(s1 - s2) # *1.0
        
    #     # diff
    #     diff_hue = abs(h2 - h1)
    #     # if diff_hue > 0.5:
    #     #     diff_hue = 1 - diff_hue
    #     diff_light = abs(l1 - l2)
    #     diff_saturation = abs(s1 - s2)

    #     # CALCULATE WEIGHTED DISTANCE
    #     hue_weight = 1
    #     light_weight = 1
    #     saturation_weight = 1

    #     dist = math.sqrt((diff_hue * hue_weight)**2 + (diff_light * light_weight)**2 + (diff_saturation * saturation_weight)**2)

    #     return dist

    # MAIN: PROCESS IMAGE
    def process_image(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "Select an image first.")
            return

        target_w = self.width_var.get()
        target_h = self.height_var.get()
        target_k = self.target_colors_var.get()

        # RESIZE & PREPARE (RGB uint8)
        img_small = self.original_image.resize((target_w, target_h), Image.NEAREST).convert("RGB")
        img_small_array = np.array(img_small)  # shape (h, w, 3), dtype=uint8
        h, w, d = img_small_array.shape
        pixels_small = img_small_array.reshape((h * w, d))

        # ----- CHUYỂN SANG LAB -----
        # skimage.color.rgb2lab expects floats in [0,1]
        pixels_small_float = pixels_small.astype(np.float32) / 255.0
        # rgb2lab can accept shape (1, N, 3) as an "image" and return (1, N, 3)
        pixels_lab = color.rgb2lab(pixels_small_float.reshape(1, -1, 3))[0]  # shape (h*w, 3)

        # Convert palette (assume self.palette_np is RGB 0-255, shape (P,3))
        palette_rgb = (self.palette_np.astype(np.float32) / 255.0)
        palette_lab = color.rgb2lab(palette_rgb.reshape(1, -1, 3))[0]  # shape (P,3)

        print(f"Clustering into {target_k} segments (in LAB space)...")

        # K-MEANS on LAB
        kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
        pixel_labels = kmeans.fit_predict(pixels_lab)
        dominant_colors_lab = kmeans.cluster_centers_  # centers in LAB

        # MAPPING: build distance matrix (clusters x palette)
        dist_matrix = np.zeros((target_k, len(self.palette_np)), dtype=np.float64)

        # Option A (default): ΔE76 ≈ Euclidean distance in LAB
        for i, center_lab in enumerate(dominant_colors_lab):
            # Broadcasting: compute distance to all palette colors quickly
            dists = np.linalg.norm(palette_lab - center_lab, axis=1)
            dist_matrix[i, :] = dists

        # --- Nếu bạn muốn dùng ΔE2000 (chính xác hơn), bỏ chú thích đoạn này ---
        # from skimage.color import deltaE_ciede2000
        # for i, center_lab in enumerate(dominant_colors_lab):
        #     # deltaE_ciede2000 expects arrays of same shape; stack center to palette shape
        #     center_stack = np.tile(center_lab[np.newaxis, :], (palette_lab.shape[0], 1))
        #     dist_matrix[i, :] = deltaE_ciede2000(center_stack, palette_lab)

        # 2. Initial assignment: mỗi cụm chọn màu palette gần nhất
        initial_assignment = np.argmin(dist_matrix, axis=1)

        # 3. Xử lý xung đột (giữ nguyên logic của bạn, chỉ lưu ý threshold bây giờ là "ΔE units")
        final_mapping = {}  # {cluster_idx: palette_idx}
        palette_usage = {}
        for cluster_idx, pal_idx in enumerate(initial_assignment):
            palette_usage.setdefault(pal_idx, []).append(cluster_idx)

        # HALLUCINATION_THRESHOLD là khác biệt ΔE tối đa chấp nhận được khi đổi màu (đơn vị ΔE)
        HALLUCINATION_THRESHOLD = 15.0

        used_palette_indices = set(palette_usage.keys())

        for pal_idx, clusters in palette_usage.items():
            if len(clusters) == 1:
                final_mapping[clusters[0]] = pal_idx
            else:
                # cluster nào gần palette này nhất sẽ "giữ" nó
                clusters.sort(key=lambda c_idx: dist_matrix[c_idx][pal_idx])
                winner = clusters[0]
                final_mapping[winner] = pal_idx

                losers = clusters[1:]
                for l_idx in losers:
                    sorted_indices = np.argsort(dist_matrix[l_idx])
                    found_new_home = False

                    original_best_dist = dist_matrix[l_idx][pal_idx]

                    for candidate_pal_idx in sorted_indices:
                        if candidate_pal_idx == pal_idx:
                            continue

                        # ưu tiên palette chưa dùng
                        if candidate_pal_idx not in used_palette_indices:
                            new_dist = dist_matrix[l_idx][candidate_pal_idx]
                            if new_dist - original_best_dist < HALLUCINATION_THRESHOLD:
                                final_mapping[l_idx] = candidate_pal_idx
                                used_palette_indices.add(candidate_pal_idx)
                                found_new_home = True
                                break

                    if not found_new_home:
                        # không tìm được palette thay thế phù hợp -> cho dùng chung
                        final_mapping[l_idx] = pal_idx

        # --- BƯỚC 4: TÁI TẠO ẢNH (dùng màu RGB từ palette ban đầu) ---
        final_pixels = np.zeros_like(pixels_small)
        for i in range(target_k):
            mask = (pixel_labels == i)
            pal_idx = final_mapping.get(i, initial_assignment[i])
            color_val = self.palette_np[pal_idx]  # kiểu uint8 RGB [0..255]
            final_pixels[mask] = color_val

        # --- THỐNG KÊ ---
        final_pixels_list = [tuple(row) for row in final_pixels]
        stats = Counter(final_pixels_list)
        print(f"Target K: {target_k} | Actual Used: {len(stats)}")

        result_array = final_pixels.reshape((int(h), int(w), 3)).astype(np.uint8)
        self.processed_image = Image.fromarray(result_array)

        self.is_showing_original = False
        self.show_preview(self.processed_image, self.zoom_scale_processed)
        self.update_stats(stats)
        print("Complete.")

    def update_stats(self, stats):
        # Delete old stats
        for widget in self.stats_content.winfo_children():
            widget.destroy()

        # Total colors
        tk.Label(self.stats_content, text=f"Total Colors: {len(stats)}", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 5))

        # From most common pixels to least
        sorted_stats = stats.most_common()

        # Stats by lines
        for color_rgb, count in sorted_stats:
            # Frame for each fow
            row_frame = tk.Frame(self.stats_content)
            row_frame.pack(fill=tk.X, pady=1, anchor="w")

            hex_color = rgb_to_hex(color_rgb)

            # color box and pixel count
            color_box = tk.Label(row_frame, bg=hex_color, width=6, height=1, relief="ridge", bd=1)
            color_box.pack(side=tk.LEFT, padx=(0, 10))
            info_label = tk.Label(row_frame, text=f"{count} pixels", font=("Consolas", 9))
            info_label.pack(side=tk.LEFT)

            # Hover to show
            def on_enter(e, label=info_label, h=hex_color):
                label.config(text=str(h), fg="blue", font=("Consolas", 9, "bold"))
            def on_leave(e, label=info_label, c=count):
                label.config(text=f"{c} pixels", fg="black", font=("Consolas", 9))
            color_box.bind("<Enter>", on_enter)
            color_box.bind("<Leave>", on_leave)


    # --- SAVE IMAGE ---
    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.processed_image.save(file_path)
                messagebox.showinfo("Success", "Image exported successfully.")






if __name__ == "__main__":
    root = tk.Tk()
    app = PixelArtTool(root)
    root.mainloop()