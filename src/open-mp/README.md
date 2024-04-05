# Tugas Kecil - Paralel Inverse Matrix dengan Open MP
Program ini merupakan program untuk melakukan eliminasi Gauss-Jordan terdistribusi secara paralel menggunakan Open MP.
## How to Run
```console
user@user:~/if3230-tucil-brother_$ make parallelMP
atau
user@user:~/if3230-tucil-brother_$ make parallelMPMacbook
```
*contoh bisa dilihat pada file makefile*

### Proses Kerja

1. Program diawali dengan alokasi memori untuk array matriks yang akan digunakan. #pragma omp parrallel for private(i) digunakan untuk mengalokasikan memori secara paralel.

2. Inissialiasi right-hand side matriks identitas. #pragma omp parallel for private(j) digunakan untuk menginisialisasi matriks secara paralel.

3. Partial Pivoting dilakukan untuk memastikan bahwa elemen diagonal matriks tidak nol. #pragma omp parallel for private(j, d) digunakan untuk melakukan partial pivoting secara paralel.

4. Reducing matriks menjadi matriks diagonal. #pragma omp parallel for private(j, d, k) digunakan untuk mengurangi matriks menjadi matriks diagonal secara paralel.

5. Reducing matriks menjadi unit matriks. #pragma omp parallel for private(j, d) digunakan untuk mengurangi matriks menjadi unit matriks secara paralel.

Pemilihan skema pembagian data dilakukan dengan membagi-bagikan baris matriks ke proses-proses yang ada sehingga setiap proses mendapat bagian matriks yang cukup untuk dilakukan perhitungan secara paralel. Skema ini memastikan bahwa pekerjaan terbagi merata di antara proses-proses yang tersedia, mengoptimalkan penggunaan sumber daya pada sistem paralel yang digunakan.