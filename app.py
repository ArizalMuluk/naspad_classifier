import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- 2. Konfigurasi Aplikasi ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/nasipad_classifier.pt' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 3. Definisi Model dan Data ---

# Daftar masakan dan deskripsinya
MASAKAN_PADANG = {
    'ayam_goreng': {'nama': 'Ayam Goreng', 'deskripsi': 'Ayam yang digoreng dengan bumbu rempah khas Padang.'},
    'ayam_pop': {'nama': 'Ayam Pop', 'deskripsi': 'Ayam yang direbus dengan bumbu kemudian digoreng hingga kulit berwarna putih keemasan.'},
    'daging_rendang': {'nama': 'Daging Rendang', 'deskripsi': 'Masakan daging sapi yang dimasak dengan santan dan rempah-rempah dalam waktu lama.'},
    'dendeng_batokok': {'nama': 'Dendeng Batokok', 'deskripsi': 'Daging sapi yang dipukul-pukul (batokok) kemudian dikeringkan dan digoreng.'},
    'gulai_ikan': {'nama': 'Gulai Ikan', 'deskripsi': 'Masakan ikan dengan kuah santan dan bumbu rempah yang kaya.'},
    'gulai_tambusu': {'nama': 'Gulai Tambusu', 'deskripsi': 'Masakan usus sapi muda (tambusu) yang dimasak dengan kuah santan dan bumbu gulai.'},
    'gulai_tunjang': {'nama': 'Gulai Tunjang', 'deskripsi': 'Masakan kaki sapi (tunjang) yang dimasak dengan kuah santan dan rempah gulai.'},
    'telur_balado': {'nama': 'Telur Balado', 'deskripsi': 'Telur rebus atau goreng yang dimasak dengan sambal balado.'},
    'telur_dadar': {'nama': 'Telur Dadar', 'deskripsi': 'Telur yang dikocok dan digoreng tipis, sering ditambah bawang merah dan cabai.'}
}
class_names = list(MASAKAN_PADANG.keys())

# Memuat arsitektur model (HARUS SAMA DENGAN SAAT TRAINING)
# Menggunakan 'weights=None' adalah cara modern dan menghilangkan UserWarning
model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Memuat bobot yang sudah dilatih
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model berhasil dimuat dari '{MODEL_PATH}' dan berjalan di device: {device}")
except FileNotFoundError:
    print(f"ERROR: File model tidak ditemukan di '{MODEL_PATH}'. Aplikasi tidak akan bisa melakukan prediksi.")
    # Anda bisa memutuskan untuk menghentikan aplikasi jika model tidak ada
    # import sys
    # sys.exit(1)

# Set model ke mode evaluasi (sangat penting!)
model.to(device)
model.eval()

# --- 4. Fungsi Helper ---

# Transformasi untuk gambar input (harus sama dengan 'val' transform saat training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prediction(image_bytes):
    """Menerima byte gambar, melakukan prediksi, dan mengembalikan hasilnya"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted_idx.item()]
        
        return {
            'class_name': predicted_class,
            'confidence': confidence.item(),
            'info': MASAKAN_PADANG[predicted_class]
        }
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return {'error': str(e)}

# --- 5. Rute Aplikasi Flask ---

@app.route('/', methods=['GET'])
def index():
    # Menampilkan halaman utama
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Simpan file untuk ditampilkan kembali di frontend
            image_bytes = file.read()
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            # Lakukan prediksi dari byte gambar
            result = get_prediction(image_bytes)
            
            if 'error' in result:
                return jsonify(result), 500
            
            # Kirim hasil dalam format JSON
            return jsonify({
                'success': True,
                'predicted_class_name': result['info']['nama'],
                'confidence': round(result['confidence'] * 100, 2),
                'deskripsi': result['info']['deskripsi'],
                'image_path': url_for('static', filename=f'uploads/{filename}')
            })
            
        except Exception as e:
            return jsonify({'error': f'Error memproses file: {str(e)}'}), 500
    
    return jsonify({'error': 'Format file tidak didukung'}), 400

# --- 6. Jalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)