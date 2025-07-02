# Modern Object Detection & Measurement

Modern bir nesne tespiti ve ölçüm uygulaması. YOLOv11 ve BirefNet modellerini kullanarak nesneleri tespit eder, segmentasyon yapar ve hassas mesafe ölçümleri gerçekleştirir.

## Özellikler

- **YOLOv11 Nesne Tespiti**: Hassas nesne tespiti
- **BirefNet Segmentasyon**: Gelişmiş şekil segmentasyonu
- **Hassas Mesafe Ölçümü**: Gerçek kontur uzunluğu hesaplama
- **Modern GUI**: PySide6 tabanlı kullanıcı dostu arayüz
- **Gerçek Zamanlı İşleme**: Webcam ve video dosyası desteği
- **Ayarlanabilir Parametreler**: Güven, kontur eşiği, köşe tespiti, ölçek

## Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (opsiyonel, CPU da kullanılabilir)

### Paket Kurulumu

```bash
pip install -r requirements.txt
```

### Model Dosyaları

YOLOv11 modelini `models/` klasörüne yerleştirin:
```
models/
└── yolov11-small-cloths.pt
```

## Kullanım

### Uygulamayı Başlatma

```bash
python modern_gui.py
```

### Arayüz Kullanımı

1. **Video Kaynağı Seçimi**:
   - "Webcam" butonu ile canlı kamera
   - "Load Video" butonu ile video dosyası

2. **Ayarlar**:
   - **Confidence**: Tespit güven eşiği (0.1-1.0)
   - **Contour Threshold**: Kontur alan eşiği (10-1000)
   - **Corner Detection**: Köşe tespit hassasiyeti (0.01-0.1)
   - **Pixel/CM Ratio**: Piksel-santimetre oranı (0.1-1.0)

3. **Ölçüm**:
   - "Calculate Measurements" butonuna basın
   - Sonuçlar hem görselde hem de sağ panelde görüntülenir

## Ölçüm Sistemi

### Mesafe Hesaplama

1. **Nesne Tespiti**: YOLOv11 ile nesne tespiti
2. **Segmentasyon**: BirefNet ile şekil segmentasyonu
3. **Kontur Bulma**: OpenCV ile kontur tespiti
4. **Köşe Tespiti**: Kontur yaklaşımı ile köşe noktaları
5. **Mesafe Hesaplama**: 
   - Köşe noktaları arası düz mesafe
   - Tüm kontur noktaları arası gerçek uzunluk
6. **Ölçeklendirme**: Piksel değerlerini cm'ye çevirme

### Görselleştirme

- **Köşe Noktaları**: Kırmızı daireler (C1, C2, ...)
- **Mesafe Değerleri**: Sarı metin ile segment mesafeleri
- **Toplam Çevre**: Gerçek kontur uzunluğu
- **Yaklaşık Değer**: Köşe noktaları arası toplam

## Teknik Detaylar

### Modeller

- **YOLOv11**: Nesne tespiti için
- **BirefNet**: Segmentasyon için
- **OpenCV**: Görüntü işleme ve kontur analizi

### Performans

- GPU kullanımı ile hızlı işleme
- Çoklu thread desteği
- Gerçek zamanlı görüntüleme

## Sorun Giderme

### WSL2'de Çalıştırma

```bash
export DISPLAY=:0
python modern_gui.py
```

### GPU Desteği

CUDA destekli kurulum için:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Yükleme Sorunları

BirefNet modeli otomatik olarak HuggingFace'den indirilir. İnternet bağlantısı gereklidir.

## Lisans

Bu proje açık kaynak kodludur.

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin 