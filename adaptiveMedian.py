# @title Implementasi Komparatif: Box vs SMF vs Simple AMF
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from skimage import metrics

# --- 1. FUNGSI PENDUKUNG (HELPER FUNCTIONS) ---
def get_image_from_path(path):
    """Memuat citra grayscale dari path lokal."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Tambahkan pemeriksaan: Jika gambar tidak berhasil dimuat, kembalikan gambar hitam sebagai placeholder
    if img is None or img.size == 0:
        print(f"Warning: Could not load image from path: {path}. Returning a black placeholder image.")
        return np.zeros((512, 512), dtype=np.uint8) # Mengembalikan citra hitam 512x512

    # Resize ke 512x512 untuk standarisasi
    img = cv2.resize(img, (512, 512))
    return img

def add_salt_and_pepper_noise(image, density):
    """
    Menambahkan Salt-and-Pepper noise pada citra.
    density: persentase noise (0.0 - 1.0)
    """
    noisy_image = image.copy()
    num_pixels = image.size

    # Jumlah noise salt (putih) dan pepper (hitam)
    num_salt = np.ceil(density * num_pixels * 0.5)
    num_pepper = np.ceil(density * num_pixels * 0.5)

    # Menambahkan Salt (255)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255

    # Menambahkan Pepper (0)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0

    return noisy_image

def calculate_psnr(original, compressed):
    """Menghitung Peak Signal-to-Noise Ratio (PSNR)."""
    return metrics.peak_signal_noise_ratio(original, compressed, data_range=255)

def calculate_mse(original, compressed):
    """Menghitung Mean Squared Error (MSE)."""
    return metrics.mean_squared_error(original, compressed)

# --- 2. IMPLEMENTASI ALGORITMA FILTER ---

def box_filter(image, kernel_size=3):
    """Filter Linear Rata-rata (Mean/Box Filter)."""
    return cv2.blur(image, (kernel_size, kernel_size))

def standard_median_filter(image, kernel_size=3):
    """Standard Median Filter (SMF)."""
    return cv2.medianBlur(image, kernel_size)

def simple_adaptive_median_filter(image, kernel_size=3):
    """
    Simple Adaptive/Switching Median Filter (Usulan Novelty).
    Logika:
    1. Deteksi: Apakah piksel 0 atau 255?
    2. Switching:
       - Jika Ya (Noise): Ganti dengan nilai Median tetangga.
       - Jika Tidak (Sehat): Biarkan nilai asli.
    """
    # Langkah 1: Hitung median untuk seluruh citra (kandidat pengganti)
    median_filtered = cv2.medianBlur(image, kernel_size)

    # Langkah 2: Buat Mask Deteksi (Logika Deteksi)
    # True jika piksel adalah 0 (pepper) atau 255 (salt)
    noise_mask = (image == 0) | (image == 255)

    # Langkah 3: Switching
    # Jika mask True (noise), ambil dari median_filtered.
    # Jika mask False (bersih), ambil dari image asli.
    # np.where adalah implementasi vektorisasi yang cepat untuk logika if-else per piksel.
    restored_image = np.where(noise_mask, median_filtered, image)

    return restored_image.astype(np.uint8)

# --- 3. EKSEKUSI PENELITIAN UTAMA ---

# A. Setup Data
image_path = "/content/citra_pepper.tiff"
original_img = get_image_from_path(image_path)


# Variasi densitas noise sesuai rencana penelitian
noise_densities = [0.1, 0.3, 0.5, 0.7, 0.9]
results = {'box': [], 'smf': [], 'amf': []}

print(f"{'Densitas':<10} | {'Metode':<10} | {'PSNR (dB)':<10} | {'MSE':<10}")
print("-" * 50)

# B. Loop Pengujian
for d in noise_densities:
    # 1. Tambahkan Noise
    noisy_img = add_salt_and_pepper_noise(original_img, d)

    # 2. Terapkan Filter
    res_box = box_filter(noisy_img)
    res_smf = standard_median_filter(noisy_img)
    res_amf = simple_adaptive_median_filter(noisy_img) # Algoritma Usulan

    # 3. Hitung Metrik
    psnr_box = calculate_psnr(original_img, res_box)
    psnr_smf = calculate_psnr(original_img, res_smf)
    psnr_amf = calculate_psnr(original_img, res_amf)

    mse_box = calculate_mse(original_img, res_box)
    mse_smf = calculate_mse(original_img, res_smf)
    mse_amf = calculate_mse(original_img, res_amf)

    # Simpan hasil untuk plotting
    results['box'].append(psnr_box)
    results['smf'].append(psnr_smf)
    results['amf'].append(psnr_amf)

    print(f"{d*100:<9.0f}% | {'Box':<10} | {psnr_box:<10.2f} | {mse_box:<10.2f}")
    print(f"{'':<10} | {'SMF':<10} | {psnr_smf:<10.2f} | {mse_smf:<10.2f}")
    print(f"{'':<10} | {'AMF (Usulan)':<10} | {psnr_amf:<10.2f} | {mse_amf:<10.2f}")
    print("-" * 50)

# --- 4. VISUALISASI HASIL (Untuk Densitas 30% dan 70% sebagai sampel) ---
sample_densities = [0.3, 0.7]
fig, axes = plt.subplots(len(sample_densities), 4, figsize=(16, 8))

for i, d in enumerate(sample_densities):
    noisy_sample = add_salt_and_pepper_noise(original_img, d)

    res_box = box_filter(noisy_sample)
    res_smf = standard_median_filter(noisy_sample)
    res_amf = simple_adaptive_median_filter(noisy_sample)

    images = [noisy_sample, res_box, res_smf, res_amf]
    titles = [f"Noisy (Density {int(d*100)}%)", "Box Filtered", "SMF Filtered", "AMF Filtered (Proposed)"]

    for j, (img, title) in enumerate(zip(images, titles)):
        axes[i, j].imshow(img, cmap='gray')
        axes[i, j].set_title(title)
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()

# --- 5. GRAFIK PERBANDINGAN PSNR ---
plt.figure(figsize=(10, 6))
plt.plot(noise_densities, results['box'], marker='o', label='Box Filter', linestyle='--')
plt.plot(noise_densities, results['smf'], marker='s', label='Standard Median (SMF)', linestyle='-.')
plt.plot(noise_densities, results['amf'], marker='^', label='Simple Adaptive (AMF)', linewidth=2.5)

plt.title('Analisis Komparatif PSNR terhadap Densitas Noise')
plt.xlabel('Densitas Noise (0.1 - 0.9)')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid(True)
plt.show()