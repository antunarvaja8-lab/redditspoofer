from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image, ImageEnhance, ImageOps
import os
import zipfile
from datetime import datetime, timedelta
import random
import shutil
import io
import string
import secrets
import numpy as np
from scipy.fftpack import dct  # para pHash

app = Flask(__name__)
app.config["ZIP_FOLDER"] = "zips"
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024
app.config["MAX_FILE_SIZE"] = 10 * 1024 * 1024
app.config["MAX_FILES"] = 100
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}

os.makedirs(app.config["ZIP_FOLDER"], exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413

def clean_old_zips():
    now = datetime.now()
    for filename in os.listdir(app.config["ZIP_FOLDER"]):
        path = os.path.join(app.config["ZIP_FOLDER"], filename)
        if os.path.isfile(path) and now - datetime.fromtimestamp(os.path.getmtime(path)) > timedelta(hours=1):
            os.remove(path)

def remove_metadata(img: Image.Image) -> Image.Image:
    data = list(img.getdata())
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(data)
    return clean_img

def generate_random_filename(extension="jpg", length=20):
    chars = string.ascii_letters + string.digits
    return f"{''.join(secrets.choice(chars) for _ in range(length))}.{extension}"

def phash(img: Image.Image, hash_size=8):
    try:
        img = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=float)
        dct_coeff = dct(dct(pixels.T, norm='ortho').T, norm='ortho')
        dctlowfreq = dct_coeff[:hash_size, :hash_size]
        med = np.median(dctlowfreq)
        return ''.join('1' if v > med else '0' for v in dctlowfreq.flat)
    except:
        return '0' * (hash_size ** 2)

def hamming_distance(h1, h2):
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def apply_transformations(img: Image.Image, params: dict):
    img = ImageOps.exif_transpose(img)  # Corregir orientación original
    orig_w, orig_h = img.size
    img = remove_metadata(img)

    # Brillo y contraste
    img = ImageEnhance.Brightness(img).enhance(params['brightness'])
    img = ImageEnhance.Contrast(img).enhance(params['contrast'])

    # Crop random + resize sutil
    crop_factor = random.uniform(0.85, 0.95)  # achica 5-15%
    crop_w = int(orig_w * crop_factor)
    crop_h = int(orig_h * crop_factor)

    # Offset random para crop (de -20% a +20% del centro)
    offset_x = random.uniform(-0.20, 0.20)
    offset_y = random.uniform(-0.20, 0.20)

    left = int((orig_w - crop_w) / 2 + offset_x * (orig_w - crop_w))
    top  = int((orig_h - crop_h) / 2 + offset_y * (orig_h - crop_h))
    right = left + crop_w
    bottom = top + crop_h

    # Asegurar que no salga fuera de límites
    left = max(0, min(left, orig_w - crop_w))
    top  = max(0, min(top, orig_h - crop_h))
    img = img.crop((left, top, right, bottom))

    # Volver al tamaño original (resize suave)
    img = img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # Mirroring si corresponde
    if params['mirror']:
        img = ImageOps.mirror(img)

    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_images():
    clean_old_zips()

    if 'images' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("images")
    if len(files) > app.config["MAX_FILES"]:
        return jsonify({"error": f"Max {app.config['MAX_FILES']} files"}), 400

    for f in files:
        if f.content_length and f.content_length > app.config["MAX_FILE_SIZE"]:
            return jsonify({"error": f"File {f.filename} too large"}), 400

    try:
        variations = min(max(int(request.form.get("variations", 1)), 1), 10)
    except:
        return jsonify({"error": "Invalid variations"}), 400

    apply_mirroring_opt = request.form.get("mirroring", "on") == "on"

    valid_files = [f for f in files if f and allowed_file(f.filename)]
    if not valid_files:
        return jsonify({"error": "No valid images"}), 400

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_code = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(5))
    zip_filename = f"spoofed_{timestamp}.zip"
    zip_path = os.path.join(app.config["ZIP_FOLDER"], zip_filename)

    hash_diffs = []
    processed = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for var in range(1, variations + 1):
            # Generar params UNA VEZ por variación - versión sutil ±10%
            brightness = random.uniform(0.90, 1.10)
            contrast = random.uniform(0.90, 1.10)
            mirror = apply_mirroring_opt and random.random() > 0.5

            params = {
                'brightness': brightness,
                'contrast': contrast,
                'mirror': mirror
                # ya no rotation ni zoom
            }

            for file in valid_files:
                try:
                    file.stream.seek(0)
                    img = Image.open(file.stream)
                    orig_phash = phash(img)

                # orig_phash = phash(img)
                
                    transformed = apply_transformations(img, params)
                    #trans_phash = phash(transformed)
                    #diff = hamming_distance(orig_phash, trans_phash)
                    #hash_diffs.append(diff)

                    ext = file.filename.rsplit('.', 1)[-1].lower()
                    output_ext = ext if ext in ['jpg', 'jpeg', 'png', 'webp'] else 'jpg'
                    fname = generate_random_filename(output_ext)
                    path_in_zip = f"var_{var}_{random_code}/{fname}"

                    buf = io.BytesIO()
                    if output_ext in ['jpg', 'jpeg']:
                        transformed.save(buf, format='JPEG', quality=95, optimize=True)  # 95 reduce tamaño ~20-30%, imperceptible
                    elif output_ext == 'png':
                        transformed.save(buf, format='PNG', optimize=True, compress_level=6)  # Compresión media lossless
                    else:  # webp
                        transformed.save(buf, format='WEBP', quality=95)  # Similar para WebP

                    zipf.writestr(path_in_zip, buf.getvalue())
                    processed += 1
                except Exception as e:
                    pass  # skip

    avg_diff = round(sum(hash_diffs) / len(hash_diffs), 1) if hash_diffs else 0

    return jsonify({
        "success": True,
        "zip_url": f"/download/{zip_filename}",
        "processed": processed,
        "variations": variations,
        "hash_diff_avg": 0,  # temporal
        "note": "pHash disabled for testing"
    })

@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(app.config["ZIP_FOLDER"], filename, as_attachment=True)
    except:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)