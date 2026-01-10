# Panduan Menjalankan Notebook (IndoNLU)

Panduan ini dibuat untuk memudahkan Anda menjalankan ulang notebook di masa depan, khususnya `examples/finetune_fnb.ipynb`.

## 1. Persiapan Awal (Prerequisites)

Pastikan Anda sudah menginstall:

- **Python** (versi 3.8 - 3.11 direkomendasikan).
- **VS Code** dengan extension **Python** dan **Jupyter**.

## 2. Membuat & Mengaktifkan Environment (Windows)

Sangat disarankan menggunakan `venv` agar library tidak konflik. Buka terminal di folder root project (`d:\coding\python\indonlu`), lalu jalankan:

### Langkah 1: Buat Virtual Environment

Jika folder `venv` belum ada:

```powershell
python -m venv venv
```

### Langkah 2: Aktifkan Environment

Setiap kali membuka VS Code baru, pastikan ini dijalankan:

```powershell
.\venv\Scripts\Activate
```

_(Tanda `(venv)` akan muncul di awal baris terminal jika berhasil)_

### Langkah 3: Install Library

```powershell
pip install -r requirements.txt
```

### Langkah 4: Setup Jupyter Kernel

Agar environment terbaca di notebook:

```powershell
pip install ipykernel
python -m ipykernel install --user --name=indonlu-env
```

## 3. Cara Menjalankan Notebook

1.  Buka file notebook (misal: `examples/finetune_fnb.ipynb`).
2.  Lihat pojok kanan atas VS Code, klik tombol kernel (biasanya tertulis "Python 3...").
3.  Pilih **Select Kernel** -> **Python Environments**.
4.  Pilih environment yang ada di folder `venv` atau `indonlu-env`.
5.  Jalankan cell secara berurutan.

## 4. Tips Penting

- **GPU vs CPU**:

  - Script ini default-nya menggunakan GPU (`model.cuda()`).
  - Jika error `AssertionError: Torch not compiled with CUDA enabled` atau komputer tidak punya GPU NVIDIA, cari baris `model = model.cuda()` dan ganti menjadi:
    ```python
    model = model.to('cpu')
    ```
    _(Note: Training di CPU akan jauh lebih lambat)_

- **Working Directory**:
  - Cell pertama di notebook sangat penting (`sys.path.append('../')`, `os.chdir('../')`). Jangan di-skip, karena ini mengatur agar notebook bisa membaca folder `utils` dan `dataset` dengan benar.
