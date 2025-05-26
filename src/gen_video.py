from PIL import Image
import imageio
import os

def resize_and_pad(img, target_height, target_width):
    """Resize to target height, then pad to target width."""
    w, h = img.size
    if h != target_height:
        new_w = int(w * (target_height / h))
        img = img.resize((new_w, target_height))
    if img.width < target_width:
        padded = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        offset = (target_width - img.width) // 2
        padded.paste(img, (offset, 0))
        return padded
    return img

def generate_initial_vs_final_panel(output_path="output/dashboard_views_compare.png"):
    views = ["top", "side", "front", "iso"]
    target_height = 400
    target_width = 1000  # for side-by-side (500 each)

    rows = []

    for view in views:
        init_path = f"output/initial_positions_{view}.png"
        final_path = f"output/registered_final_{view}.png"

        if not os.path.exists(init_path) or not os.path.exists(final_path):
            print(f"Skipping {view} (missing file)")
            continue

        img_init = resize_and_pad(Image.open(init_path), target_height, target_width // 2)
        img_final = resize_and_pad(Image.open(final_path), target_height, target_width // 2)

        row = Image.new("RGB", (target_width, target_height))
        row.paste(img_init, (0, 0))
        row.paste(img_final, (target_width // 2, 0))
        rows.append(row)

    if not rows:
        print("No views found. Skipping composite image.")
        return None

    # Combine rows vertically
    full_height = target_height * len(rows)
    final_image = Image.new("RGB", (target_width, full_height))
    for i, row in enumerate(rows):
        final_image.paste(row, (0, i * target_height))

    final_image.save(output_path)
    print(f"✅ Saved comparison panel to: {output_path}")
    return output_path


def make_registration_gif_and_video(static_panel_path, output_gif="dashboard_compare.gif", output_mp4="dashboard_compare.mp4"):
    # Load static top-left panel (initial iso)
    static_panel = Image.open(static_panel_path)

    frames = []

    for i in range(1, 51):
        frame_path = f"output/registered_observations_{i}.png"
        if not os.path.exists(frame_path):
            print(f"Skipping frame {frame_path}")
            continue

        reg_frame = Image.open(frame_path)

        # Resize reg_frame to match static_panel height
        target_height = static_panel.height
        if reg_frame.height != target_height:
            new_w = int(reg_frame.width * (target_height / reg_frame.height))
            reg_frame = reg_frame.resize((new_w, target_height))

        # Combine side-by-side
        combined = Image.new("RGB", (static_panel.width + reg_frame.width, target_height), (255, 255, 255))
        combined.paste(static_panel, (0, 0))
        combined.paste(reg_frame, (static_panel.width, 0))

        frames.append(combined)

    if not frames:
        print("❌ No frames created for video/GIF.")
        return

    # Save as GIF
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=300, loop=0)
    print(f"✅ Saved {output_gif}")

    # Save as MP4
    imageio.mimsave(output_mp4, [imageio.v3.imread(f) for f in frames], fps=5)
    print(f"✅ Saved {output_mp4}")


# === Run the script ===
panel_path = generate_initial_vs_final_panel()
if panel_path:
    make_registration_gif_and_video(panel_path)
