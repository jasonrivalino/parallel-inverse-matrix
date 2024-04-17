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
Setiap baris matrix didistribusikan ke 4 blok, dimana setiap blok terdiri dari n/2 thread. Alasan pemilihan skema ini adalah banyak thread maksimun dalam satu blok adalah 1024 thread, dan pada test case terbesar yang diberikan adalah matriks dengan ukuran 2048x2048, dimana setelah dijadikan matriks augmented ukuran menjadi 2048x4096, sehingga 1 baris akan membutuhkan 4 blok yang berisi 4096 thread, agar 1 thread hanya perlu menghitung 1 elemen matriks augmented.