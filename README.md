Penelitian ini mengembangkan sistem deteksi objek dan estimasi kedalaman
untuk mendukung koordinat gerak robot arm Nachi MZ04 pada proses pick and
place. Objek yang digunakan adalah roset telepon yang tersusun tidak beraturan
sehingga diperlukan metode yang mampu mendeteksi posisi sekaligus
memperkirakan kedalaman (Z) secara akurat.
Deteksi objek dilakukan menggunakan algoritma YOLOv8 pada citra
kamera monocular, sedangkan estimasi kedalaman diperoleh melalui Fuzzy
Inference System (FIS) dengan input panjang diagonal dan rasio aspek bounding
box. Data dikumpulkan pada kondisi pencahayaan terang (666 lux), redup (42 lux
dan 1 lux), serta variasi objek bertumpuk hingga empat lapis dan diletakkan di
pojok wadah.
Hasil pengujian menunjukkan YOLOv8 mendeteksi objek dengan akurasi
100% pada kondisi tidak bertumpuk, namun menurun menjadi 50%, 33,3%, dan
25% pada kondisi bertumpuk dua hingga empat lapis. Penurunan terjadi karena
YOLOv8 hanya mengenali lapisan paling atas, sementara objek di bawah tidak
terdeteksi. Estimasi kedalaman dengan FIS menghasilkan error rata-rata 0,3–0,5
cm, meningkat hingga 0,8 cm pada pencahayaan redup, tetapi tetap di bawah
toleransi < 1 cm. Integrasi YOLOv8 dan FIS terbukti efektif untuk estimasi
koordinat robot serta potensial diterapkan langsung pada sistem kontrol secara real
berikut adalah vidio demonya : https://youtu.be/W3jWEPY7g68

time.![Poster Tugas Akhir_page-0001](https://github.com/user-attachments/assets/18210e23-91e5-4076-8fc1-2cf288603598)

