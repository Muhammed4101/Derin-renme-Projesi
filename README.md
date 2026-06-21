# Evrensel Derin Öğrenme Mimarileri ile Yaprak Üzerinden Bitki Hastalığı Tespiti: Benchmark Analizi

[cite_start]Bu proje; tarımsal verimliliği artırmak amacıyla bitki yapraklarındaki hastalıkları erken evrede tespit etmeyi hedefleyen uçtan uca bir bilgisayarlı görü (computer vision) çalışmasıdır[cite: 5]. [cite_start]Proje kapsamında, literatürde standart kabul edilen 4 farklı derin öğrenme mimarisi (VGG16, ResNet-50, MobileNetV2, EfficientNetB0) hem sıfırdan (scratch) eğitilmiş hem de ImageNet ağırlıklarıyla transfer öğrenme (transfer learning) süzgecinden geçirilerek süre ve başarı odaklı karşılaştırmalı analize (benchmark) tabi tutulmuştur[cite: 6, 7, 8].

---

## 📑 Yönetici Özeti (Executive Summary)

* [cite_start]**Veri Kümesi Gücü:** Projede 38 farklı sınıfa ait 54.305 adet yüksek çözünürlüklü yaprak görseli içeren, bilgisayarlı görünün bitki patolojisindeki en güvenilir veri setlerinden biri olan **PlantVillage** kullanılmıştır[cite: 6].
* [cite_start]**Temel Problem (Neden Sıfırdan Eğitim Değil?):** Sıfırdan yapılan eğitimlerde modeller, veri kümesindeki temel geometrik özellikleri (kenar, doku, yeşil ve kahverengi renk tonları) sıfırdan öğrenmeye çalışırken devasa bir zaman kaybetmiş, lokal minimumlara takılmış ve yüksek parametre kapasiteleri yüzünden hızla ezberlemeye (overfitting) yönelmiştir[cite: 46].
* **Transfer Öğrenmenin Gücü:** ImageNet üzerinde önceden eğitilmiş ağırlıkların kullanılması, genel öznitelik çıkarım yeteneğini projeye doğrudan aktarmıştır. [cite_start]Bu sayede modellerin doğruluk oranları %38-%76 bandından doğrudan **%90 ve üzerine** fırlamış, epoch başına işlem yükü ise ortalama 3 ila 4 kat azalarak donanım maliyetlerini minimize etmiştir[cite: 12, 16, 22, 25, 32, 36, 47].
* [cite_start]**Projenin Şampiyonu:** Hem hız hem de doğruluk dengesinde (Speed-Accuracy Trade-off) rakiplerine fark atan ve en düşük doğrulama kaybını (`0.2475`) sunan **EfficientNetB0**, projenin en başarılı modeli seçilmiştir[cite: 43, 44].

---

## 🛠️ Mimari Değerlendirmeler ve Teknik Detaylar

### 1. VGG16 Mimarisi
[cite_start]Klasik, ardışık (sequential) konvolüsyon katmanlarından oluşan ve derin öğrenmenin temel taşlarından sayılan VGG16, bu projede doğrusal öğrenme kapasitesi ve parametre yükü açısından test edilmiştir[cite: 6, 18].

* [cite_start]**Sıfırdan Eğitim (Scratch):** Eğitim setindeki ezberleme oranı `%93.58` gibi yüksek bir seviyeye çıksa da, validation doğruluğu `%76.36` seviyesinde kalmıştır[cite: 12, 13]. [cite_start]Bu durum, modelin `1.4319` gibi yüksek bir doğrulama kaybı (val_loss) ile feci şekilde overfitting'e düştüğünü kanıtlamaktadır[cite: 14]. [cite_start]Epoch süresi ise hantal yapısından dolayı `~10.5 dakikayı` bulmuştur[cite: 12, 14].
* [cite_start]**Transfer Öğrenme (ImageNet):** Önceden eğitilmiş ağırlıklar hantal yapının ezberleme eğilimini tamamen baskılamış, modeli kararlı bir çizgiye çekerek validation başarısını `%87.10`'a yükseltmiştir[cite: 15, 16, 17]. [cite_start]Dondurulan katmanlar sayesinde epoch süresi 2.5 kat hızlanarak `~4.3 dakikaya` düşmüştür[cite: 16, 17].
* [cite_start]**Teknik Yorum:** VGG16, modern mimarilerin arkasındaki "skip connection" veya "depthwise separable convolution" gibi yapılardan yoksun olduğu için çok yüksek parametre barındırır[cite: 18, 23, 34, 54]. [cite_start]Bu durum transfer öğrenmede bile dosya boyutunun çok büyük kalmasına ve MobileNetV2 gibi hafif modellere göre yavaş kalmasına neden olur[cite: 18, 54].

### 2. ResNet-50 Mimarisi
[cite_start]Derin ağlarda yaşanan gradyan sönümlenmesi (vanishing gradient) problemini "Artık Bağlantılar" (Skip Connections / Residual Blocks) ile çözen 50 katmanlı gelişmiş bir mimaridir[cite: 23, 27].

* [cite_start]**Sıfırdan Eğitim (Scratch):** Skip connection yapısı sayesinde eğitim kaybını çok hızlı düşürmesine rağmen, derinlik PlantVillage gibi spesifik bir veri setinde sıfırdan eğitildiğinde dezavantaja dönüşmüş; model henüz 4. epochta ezberlemeye başlayarak `%60.69` validation başarısında çakılmıştır[cite: 6, 21, 22, 24]. [cite_start]`~13.3 dakika` epoch süresiyle projenin en yavaş eğitime sahip modeli olmuştur[cite: 22, 24].
* [cite_start]**Transfer Öğrenme (ImageNet):** ResNet-50 transfer öğrenmeyle birlikte adeta küllerinden doğmuş, sıfırdan eğitime kıyasla 3.4 kat hızlanarak `~3.9 dakikaya` gerilemiş ve `%90.09` validation doğruluğu ile %90 barajını aşan ilk model olmuştur[cite: 25, 26].
* [cite_start]**Teknik Yorum:** Ağın derinlik kapasitesi öznitelik çıkarmada çok başarılıdır ancak katman sayısının fazlalığı ve parametre yoğunluğu, bu modelin kenar cihazlarda (Edge AI), mikrodenetleyicilerde veya mobil platformlarda çalıştırılmasını zorlaştıracak bir hesaplama yükü doğurmaktadır[cite: 27].

### 3. MobileNetV2 Mimarisi
Mobil ve gömülü sistemler için optimize edilmiş, geleneksel konvolüsyon yerine "Derinlemesine Ayrılabilir Konvolüsyon" (Depthwise Separable Convolution) kullanan, parametre dostu ve ters çevrilmiş artık blok (inverted residual) yapısına sahip bir mimaridir.

* [cite_start]**Sıfırdan Eğitim (Scratch):** Modelin dar ve derin tasarımı, PlantVillage veri setini sıfırdan beslediğimizde ağır bir gradyan kararsızlığına yol açmış ve ilk 4 epoch boyunca rastgele tahmin çizgisinde (`%10` doğrulukta) kilitli kalmıştır[cite: 6, 32, 34]. [cite_start]Neticede yalnızca `%38.14` validation başarısı yakalayabilmiştir[cite: 32].
* [cite_start]**Transfer Öğrenme (ImageNet):** Ağırlıklar hazır geldiğinde model olağanüstü bir sıçrama yapmış, epoch süresini `~1.6 dakikaya (99 saniye)` indirerek **projenin en hızlı eğitime sahip ağı** olmuştur[cite: 35, 36]. [cite_start]Validation doğruluğu ise `%87.40` gibi oldukça rekabetçi bir seviyeye ulaşmıştır[cite: 36].
* [cite_start]**Teknik Yorum:** MobileNetV2, eğitim ve doğrulama kayıplarının kafa kafaya gitmesiyle projenin en kararlı ve varyansı en düşük modelidir[cite: 36]. [cite_start]Tepe noktadaki doğruluğu EfficientNetB0'ın bir tık gerisinde kalsa da donanım/performans verimliliğinde liderdir[cite: 36, 37].

### 4. EfficientNetB0 Mimarisi
Ağ genişliğini, derinliğini ve giriş çözünürlüğünü birleşik bir katsayıyla (Compound Scaling) dengeli şekilde ölçekleyen, çağdaş ve son derece efektif bir derin öğrenme mimarisidir.

* [cite_start]**Sıfırdan Eğitim (Scratch):** Gelişmiş yapısına rağmen sıfırdan eğitimde zayıf öznitelik üretmiş, kararsız ve çok dalgalı bir loss grafiği çizerek erken aşamada ezberlemeye geçmiştir (`%49.02` validation doğruluğu, `~8.6 dakika` epoch süresi)[cite: 40, 42].
* [cite_start]**Transfer Öğrenme (ImageNet):** ImageNet desteğiyle model **%91.81 validation doğruluğuna** ulaşarak projenin mutlak şampiyonu olmuştur[cite: 43, 44]. [cite_start]Üstelik `0.2475` gibi projenin en düşük hata (loss) oranını sıfır ezberleme (overfitting) ve `~2.2 dakika` gibi muazzam bir epoch süresi eşliğinde sunmuştur[cite: 43, 44].
* **Teknik Yorum:** EfficientNetB0, matematiksel optimizasyonun doruk noktasıdır. [cite_start]Parametreleri bütçeli kullanırken doğruluğu maksimuma çıkarma yeteneği, onu büyük ölçekli ve yüksek doğruluk gerektiren sunucu sistemleri için rakipsiz kılmaktadır[cite: 44, 50].

---

## 📉 Diğer Modelleri Eleme Nedenlerimiz (Mühendislik Kararları)

1.  [cite_start]**VGG16 Neden Elendi?:** Günümüz standartlarına göre aşırı yüksek parametre yüküne sahiptir[cite: 14, 54]. [cite_start]Transfer öğrenmede dahi MobileNetV2'den çok daha yavaş kalması ve buna rağmen daha düşük doğruluk üretmesi mimariyi tamamen verimsiz kılmaktadır[cite: 54].
2.  [cite_start]**ResNet-50 Neden Elendi?:** `%90.09` gibi başarılı bir doğruluğa ulaşmış olsa da, EfficientNetB0 mimarisinin hem çok daha yüksek doğruluk (`%91.81`) sunması hem de ResNet-50'ye kıyasla **yarı yarıya daha hızlı eğitilmesi (3.9 dk vs 2.2 dk)** ResNet-50'nin rasyonelliğini kaybettirmiştir[cite: 25, 43, 55].

---

## 🎯 Dağıtım (Deployment) Stratejisi ve Sonuç

[cite_start]Elde edilen benchmark sonuçlarına göre gerçek hayat senaryolarında iki farklı canlıya alım (deployment) stratejisi önerilmektedir[cite: 8, 48]:

* [cite_start]**Edge AI & Mobil Entegrasyon (Drone, Otonom Robot, Akıllı Telefon):** Saniyedeki kare sayısı (FPS), düşük RAM tüketimi ve `99 saniyelik` rekor epoch hızı göz önüne alındığında, tarla üzerinde anlık tarama yapacak drone veya mobil uygulamalarda **MobileNetV2** konuşlandırılmalıdır[cite: 36, 49].
* [cite_start]**Bulut Tabanlı & Merkezi Analiz (Sunucu Sistemleri):** Çiftçilerin fotoğrafları bir merkeze yüklediği, hızdan ziyade milimetrik teşhisin ve en düşük hata payının hedeflendiği sunucu senaryolarında, projenin şampiyonu olan **EfficientNetB0** mimarisi kullanılmalıdır[cite: 44, 50].# Derin-renme-Projesi
