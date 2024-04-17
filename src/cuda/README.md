# Tugas Kecil - Paralel Inverse Matrix dengan CUDA
Program ini merupakan program untuk melakukan eliminasi Gauss-Jordan terdistribusi secara paralel menggunakan CUDA (Compute Unified Device Architecture).

## Cara Kerja Program:
1. Matriks menerima input dari file txt.
2. Memori dialokasikan untuk matriks baik di host (CPU) maupun di perangkat (GPU) menggunakan new dan cudaMalloc.
3. Data matriks disalin dari host ke perangkat menggunakan cudaMemcpy.
4. Terdapat beberapa kernel yang dieksekusi secara paralel pada GPU, yaitu kernel `makeRightHandSideIdentity`,`reduceToDiagonal`, dan `reduceToUnitMatrix`.
5. `cudaDeviceSynchronize` dipanggil untuk sinkronisasi untuk memastikan bahwa semua kernel selesai dieksekusi sebelum melanjutkan ke langkah berikutnya.
6. Setelah komputasi selesai, data matriks disalin dari perangkat ke host menggunakan cudaMemcpy.
7. Memori yang dialokasikan baik di host maupun di perangkat didealokasikan menggunakan delete[] dan cudaFree.

## Skema Distribusi Data:
Setiap baris matrix didistribusikan ke 4 blok, dimana setiap blok terdiri dari setengah dari n/2 thread. Alasan pemilihan skema ini adalah jika terdapat matrix dengan ukuran 2048 x 2048, maka setiap blok akan memiliki 1024 thread, sehingga setiap blok akan memiliki 1024 thread yang akan melakukan perhitungan secara paralel. Hal ini akan dapat mempercepat proses perhitungan.