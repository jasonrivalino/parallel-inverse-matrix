# Tugas Kecil - Parallel Inverse Matrix IF3230 Sistem Paralel dan Terdistribusi
> Tugas Kecil 1 - Parallel Inverse Matrix dengan OpenMPI

> Tugas Kecil 2 - Parallel Inverse Matrix dengan OpenMP

> Tugas Kecil 3 - Parallel Inverse Matrix dengan Cuda

## Anggota Kelompok
<table>
    <tr>
        <td>No.</td>
        <td>Nama</td>
        <td>NIM</td>
    </tr>
    <tr>
        <td>1.</td>
        <td>Bintang Hijriawan</td>
        <td>13521003</td>
    </tr>
    <tr>
        <td>2.</td>
        <td>Jason Rivalino</td>
        <td>13521008</td>
    </tr>
    <tr>
        <td>3.</td>
        <td>Muhammad Salman Hakim Alfarisi</td>
        <td>13521010</td>
    </tr>
</table>


## Table of Contents
* [Deskripsi Singkat](#deskripsi-singkat)
* [Struktur File](#struktur-file)
* [Requirements](#requirements)
* [Acknowledgements](#acknowledgements)

## Deskripsi Singkat
Tugas ini merupakan program untuk menerapkan operasi perhitungan inverse terhadap Matrix yang dilakukan dengan proses parallelisasi. Implementasi dilakukan dengan memanfaatkan OpenMPI, OpenMP,dan Cuda. Untuk OpenMPI dan OpenMP, program dibuat dengan menggunakan bahasa C++ dan untuk Cuda dengan menggunakan Google Colab dengan file .ipnyb untuk compile kedalam bentuk program Cuda dengan format .cu . Adapun untuk proses dalam tiap program masing-masing, bisa dengan melihat dokumentasi ReadMe yang ada pada tiap folder.

## Struktur File
```
📦if3230-tucil-brother_
 ┣ 📂.vscode
 ┣ 📂bin
 ┃ ┣ 📜.gitignore
 ┣ 📂src
 ┃ ┣ 📂cuda
 ┃ ┃ ┣ 📜.gitignore
 ┃ ┃ ┣ 📜cuda.cu
 ┃ ┃ ┣ 📜cuda_brother.ipynb
 ┃ ┃ ┗ 📜README.md
 ┃ ┣ 📂open-mp
 ┃ ┃ ┣ 📜.gitignore
 ┃ ┃ ┣ 📜parallel.cpp
 ┃ ┃ ┗ 📜README.md
 ┃ ┣ 📂open-mpi
 ┃ ┃ ┣ 📜.gitignore
 ┃ ┃ ┣ 📜parallel.cpp
 ┃ ┃ ┗ 📜README.md
 ┃ ┣ 📂sample
 ┃ ┃ ┣ 📜cuda.cu
 ┃ ┃ ┣ 📜cuda_colab.ipynb
 ┃ ┃ ┣ 📜mp.c
 ┃ ┃ ┗ 📜mpi.c
 ┃ ┗ 📂serial
 ┃ ┃ ┗ 📜serial.cpp
 ┣ 📂test_cases
 ┃ ┣ 📜1024.txt
 ┃ ┣ 📜128.txt
 ┃ ┣ 📜2048.txt
 ┃ ┣ 📜256.txt
 ┃ ┣ 📜32.txt
 ┃ ┣ 📜512.txt
 ┃ ┗ 📜64.txt
 ┣ 📜makefile
 ┗ 📜README.md
```
 
## Requirements
1. Visual Studio Code
2. Windows Subsystem For Linux (WSL)
3. Google Colab
   
## Acknowledgements
- Tuhan Yang Maha Esa
- Dosen Mata Kuliah Sistem Parallel dan Terdistribusi IF3230
- Kakak-Kakak Asisten Mata Kuliah Sistem Parallel dan Terdistribusi IF3230
