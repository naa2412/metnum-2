# Prediksi Potensi Hujan: Gradient Descent vs Steepest Descent

**Azhar Maulana** — 24/533487/PA/22582

Implementasi dua metode optimasi berbasis gradien untuk melatih model regresi linear dalam memprediksi potensi hujan berdasarkan data suhu udara, kecepatan angin, dan kelembapan relatif.

## Dataset

| Data | Suhu (°C) | Angin (km/h) | Kelembapan (%) | Potensi Hujan (%) |
|------|-----------|--------------|----------------|-------------------|
| 1    | 30        | 9            | 90             | 92                |
| 2    | 20        | 9            | 80             | 70                |
| 3    | 25        | 3            | 90             | 85                |
| Baru | 27        | 6            | 85             | ?                 |

## Model

Model yang digunakan adalah regresi linear dengan bias:

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

Parameter optimal $\mathbf{w}$ dan $b$ dicari dengan meminimalkan fungsi loss **Sum of Squared Errors (SSE)**:

$$f(\mathbf{w}, b) = \sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

## Metode

### Preprocessing
Data dinormalisasi menggunakan **Min-Max Scaling** ke rentang $[0, 1]$ sebelum proses optimasi, kemudian denormalisasi untuk mengembalikan prediksi ke skala asli.

### Gradient Descent (GD)
Parameter diperbarui setiap iterasi dengan bergerak berlawanan arah gradien sejauh *learning rate* $\alpha$ yang konstan ($\alpha = 0.1$):

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \cdot 2\mathbf{X}^\top \mathbf{e}^{(t)}$$

### Steepest Descent (SD)
Sama seperti GD, namun nilai $\alpha$ dihitung secara analitik setiap iterasi (*exact line search*) sehingga setiap langkah memberikan penurunan loss maksimal pada arah tersebut:

$$\alpha^* = -\frac{\mathbf{j}^\top \mathbf{e}}{\mathbf{j}^\top \mathbf{j}}, \quad \mathbf{j} = \mathbf{X}\mathbf{d}_w + d_b\mathbf{1}$$

## Hasil

| Metrik | Gradient Descent | Steepest Descent |
|--------|-----------------|-----------------|
| $w_1$ | 0.5272727273 | 0.5272727273 |
| $w_2$ | 0.0545454545 | 0.0545454545 |
| $w_3$ | 0.4727272727 | 0.4727272727 |
| $b$ | -0.0545454532 | -0.0545454542 |
| Loss akhir | $1.22 \times 10^{-18}$ | $7.89 \times 10^{-20}$ |
| Jumlah iterasi | **283** | **87** |
| Prediksi | **82.72%** | **82.72%** |

## Kesimpulan

Kedua metode menghasilkan prediksi yang sama (**82.72%**) karena fungsi SSE bersifat kuadratik konveks dengan satu minimum global. Perbedaannya terletak pada efisiensi: SD konvergen 3× lebih cepat (87 vs 283 iterasi) karena $\alpha^*$ dihitung optimal setiap langkah, bukan ditetapkan secara manual.

