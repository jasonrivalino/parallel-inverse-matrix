# Tugas Kecil - Paralel Inverse Matrix dengan Open MPI
Program ini merupakan program untuk melakukan eliminasi Gauss-Jordan terdistribusi secara paralel menggunakan MPI (Message Passing Interface).
## How to Run
```console
user@user:~/if3230-tucil-brother_$ mpic++ -o $(path_to_output_compile) $(path_to_source_code)
user@user:~/if3230-tucil-brother_$ mpirun -n $(num_of_process) $(path_to_output) < $(path_to_input_text) > $(path_to_output_text)
```
*contoh bisa dilihat pada file makefile*

### Proses Kerja

1. Program diawali dengan inisialisasi MPI menggunakan `MPI_Init()` dan pengambilan informasi rank dan ukuran komunikator dengan `MPI_Comm_rank()` dan `MPI_Comm_size()`.

2. Input matriks dilakukan pada proses dengan rank 0. Matriks ini kemudian disiapkan untuk dimodifikasi menjadi matriks identitas pada right-hand side-nya.

3. Broadcast ukuran matriks kemudian disebarkan kepada semua proses menggunakan `MPI_Bcast()`.

4. Data matriks kemudian di-flatten (flatMat) dan dibagi-bagikan kepada proses-proses yang ada. Fungsi `distributeRow()` digunakan untuk menghitung jumlah elemen yang akan diterima oleh setiap proses.

5. Setiap proses melakukan iterasi untuk melakukan eliminasi Gauss-Jordan terhadap bagian matriks yang diterimanya.

6. Setelah setiap iterasi eliminasi, hasilnya digabungkan kembali menggunakan `MPI_Allgather()` sehingga semua proses memiliki akses ke seluruh matriks yang telah dimodifikasi.

7. Proses dengan rank 0 mencetak hasil matriks setelah eliminasi dan menghitung otal waktu yang dibutuhkan.

8. Setelah semua proses selesai, memori yang dialokasikan dibebaskan dan MPI ditutup dengan `MPI_Finalize()`.

Pemilihan skema pembagian data dilakukan dengan membagi-bagikan baris matriks ke proses-proses yang ada sehingga setiap proses mendapat bagian matriks yang cukup untuk dilakukan perhitungan secara paralel. Skema ini memastikan bahwa pekerjaan terbagi merata di antara proses-proses yang tersedia, mengoptimalkan penggunaan sumber daya pada sistem paralel yang digunakan.