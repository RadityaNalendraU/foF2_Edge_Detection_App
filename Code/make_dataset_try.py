import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import os

# Paths (langsung ke folder yang berisi file)
img_root = Path("C:/Users/RNU/Documents/Kerja Praktek/MY_raw_DATA/raw_data/2020/FTIF_LTPMP-Feb-2020")
csv_root = Path("C:/Users/RNU/Documents/Kerja Praktek/MY_raw_DATA/txt_data/2020/FTIF_LTPMP-Feb-2020/fmin")
output_img_root = Path("C:/Users/RNU/Documents/Kerja Praktek/MY_raw_DATA/OUTPUT/img_trace_fmin/2020/FTIF_LTPMP-Feb-2020")
output_mask_root = Path("C:/Users/RNU/Documents/Kerja Praktek/MY_raw_DATA/OUTPUT/img_mask_fmin/2020/FTIF_LTPMP-Feb-2020")

# ROI Coordinates
roi_x_min = 99
roi_x_max = 700
roi_y_min = 49.5
roi_y_max = 585.5

def process_image_and_csv(img_path, csv_path, output_img_path, output_mask_path):
    try:
        image = Image.open(img_path)
        img_width, img_height = image.size
        image_np = np.array(image)

        # baca CSV
        data = pd.read_csv(csv_path)
        foF2_data = data[data['Parameter'] == 'foF2']
        fmin_data = data[data['Parameter'] == 'fmin']

        # plotting
        fig = plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image_np)
        ax.axis('off')

        roi_ax = fig.add_axes([
            roi_x_min / img_width,
            1 - roi_y_max / img_height,
            (roi_x_max - roi_x_min) / img_width,
            (roi_y_max - roi_y_min) / img_height
        ])
        roi_ax.set_facecolor((0, 0, 0, 0))
        roi_ax.plot(foF2_data['JamDec'], foF2_data['Nilai'], 'ro', markersize=2) # 'ro' = red circle marker
        roi_ax.plot(fmin_data['JamDec'], fmin_data['Nilai'], 'ro', markersize=2) # 'wo' = white circle marker
        roi_ax.set_xlim(0, 24)
        roi_ax.set_ylim(0, 20)
        roi_ax.axis('off')

        # simpan hasil overlay
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_img_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        # bikin mask merah
        overlay_image = Image.open(output_img_path).convert("RGB")
        overlay_np = np.array(overlay_image)
        red_mask = np.all(overlay_np == [255, 0, 0], axis=-1)
        mask_img = (red_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_img)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_pil.save(output_mask_path)


        print(f"✅ Processed: {img_path.name}")

    except Exception as e:
        print(f"❌ Error processing {img_path.name}: {e}")

# ============================
# MAIN LOOP TANPA SUBFOLDER
# ============================
png_files = list(img_root.glob("*.png"))
print(f"📂 Folder gambar: {img_root}")
print(f"   Ditemukan {len(png_files)} file PNG")

for img_path in png_files:
    print(f"👉 File gambar: {img_path.name}")

    # Ekstrak bagian tanggal dari nama file
    parts = img_path.stem.split("-")
    if len(parts) >= 3:
        date_part = "-".join(parts[-3:]).replace(" ", "")
        matching_csv = list(csv_root.glob(f"*{date_part}*.csv"))
        print(f"   🔗 Cari CSV di {csv_root}, pola: *{date_part}*.csv")
        print(f"   ➡️ Ditemukan: {matching_csv}")
    else:
        matching_csv = []
        print("   ⚠️ Nama file tidak cocok format (kurang tanda - )")

    if matching_csv:
        csv_path = matching_csv[0]
        out_img_path = output_img_root / f"{img_path.stem}.png"
        out_mask_path = output_mask_root / f"{img_path.stem}.png"
        process_image_and_csv(img_path, csv_path, out_img_path, out_mask_path)
    else:
        print(f"⚠️ CSV not found for {img_path.name}")

print("\n✅ All processing complete.")
