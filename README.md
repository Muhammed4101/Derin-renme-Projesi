# Evrensel Derin Öğrenme Mimarileri ile Yaprak Üzerinden Bitki Hastalığı Tespiti: Benchmark Analizi

Bu proje; tarımsal verimliliği artırmak amacıyla bitki yapraklarındaki hastalıkları erken evrede tespit etmeyi hedefleyen uçtan uca bir bilgisayarlı görü (computer vision) çalışmasıdır. Proje kapsamında, literatürde standart kabul edilen 4 farklı derin öğrenme mimarisi (VGG16, ResNet-50, MobileNetV2, EfficientNetB0) hem sıfırdan (scratch) eğitilmiş hem de ImageNet ağırlıklarıyla transfer öğrenme (transfer learning) süzgecinden geçirilerek süre ve başarı odaklı karşılaştırmalı analize (benchmark) tabi verilmiştir.

---

## 📑 Yönetici Özeti (Executive Summary)

* **Veri Kümesi Gücü:** Projede 38 farklı sınıfa ait 54.305 adet yüksek çözünürlüklü yaprak görseli içeren, bilgisayarlı görünün bitki patolojisindeki en güvenilir veri setlerinden biri olan **PlantVillage** kullanılmıştır.
* **Temel Problem (Neden Sıfırdan Eğitim Değil?):** Sıfırdan yapılan eğitimlerde modeller, veri kümesindeki temel geometrik özellikleri (kenar, doku, yeşil ve kahverengi renk tonları) sıfırdan öğrenmeye çalışırken devasa bir zaman kaybetmiş, lokal minimumlara takılmış ve yüksek parametre kapasiteleri yüzünden hızla ezberlemeye (overfitting) yönelmiştir.
* **Transfer Öğrenmenin Gücü:** ImageNet üzerinde önceden eğitilmiş ağırlıkların kullanılması, genel öznitelik çıkarım yeteneğini projeye doğrudan aktarmıştır. Bu sayede modellerin doğruluk oranları %38-%76 bandından doğrudan **%90 ve üzerine** fırlamış, epoch başına işlem yükü ise ortalama 3 ila 4 kat azalarak donanım maliyetlerini minimize etmiştir.
* **Projenin Şampiyonu:** Hem hız hem de doğruluk dengesinde (Speed-Accuracy Trade-off) rakiplerine fark atan ve en düşük doğrulama kaybını (`0.2475`) sunan **EfficientNetB0**, projenin en başarılı modeli seçilmiştir.

---

## 🛠️ Mimari Değerlendirmeler ve Teknik Detaylar

### 1. VGG16 Mimarisi
Klasik, ardışık (sequential) konvolüsyon katmanlarından oluşan ve derin öğrenmenin temel taşlarından sayılan VGG16, bu projede doğrusal öğrenme kapasitesi ve parametre yükü açısından test edilmiştir.

* **Sıfırdan Eğitim (Scratch):** Eğitim setindeki ezberleme oranı `%93.58` gibi yüksek bir seviyeye çıksa da, validation doğruluğu `%76.36` seviyesinde kalmıştır. Bu durum, modelin `1.4319` gibi yüksek bir doğrulama kaybı (val_loss) ile feci şekilde overfitting'e düştüğünü kanıtlamaktadır. Epoch süresi ise hantal yapısından dolayı `~10.5 dakikayı` bulmuştur.
* **Transfer Öğrenme (ImageNet):** Önceden eğitilmiş ağırlıklar hantal yapının ezberleme eğilimini tamamen baskılamış, modeli kararlı bir çizgiye çekerek validation başarısını `%87.10`'u yükseltmiştir. Dondurulan katmanlar sayesinde epoch süresi 2.5 kat hızlanarak `~4.3 dakikaya` düşmüştür.
* **Teknik Yorum:** VGG16, modern mimarilerin arkasındaki "skip connection" veya "depthwise separable convolution" gibi yapılardan yoksun olduğu için çok yüksek parametre barındırır. Bu durum transfer öğrenmede bile dosya boyutunun çok büyük kalmasına ve MobileNetV2 gibi hafif modellere göre yavaş kalmasına neden olur.

### 2. ResNet-50 Mimarisi
Derin ağlarda yaşanan gradyan sönümlenmesi (vanishing gradient) problemini "Artık Bağlantılar" (Skip Connections / Residual Blocks) ile çözen 50 katmanlı gelişmiş bir mimaridir.

* **Sıfırdan Eğitim (Scratch):** Skip connection yapısı sayesinde eğitim kaybını çok hızlı düşürmesine rağmen, derinlik PlantVillage gibi spesifik bir veri setinde sıfırdan eğitildiğinde dezavantaja dönüşmüş; model henüz 4. epochta ezberlemeye başlayarak `%60.69` validation başarısında çakılmıştır. `~13.3 dakika` epoch süresiyle projenin en yavaş eğitime sahip modeli olmuştur.
* **Transfer Öğrenme (ImageNet):** ResNet-50 transfer öğrenmeyle birlikte adeta küllerinden doğmuş, sıfırdan eğitime kıyasla 3.4 kat hızlanarak `~3.9 dakikaya` gerilemiş ve `%90.09` validation doğruluğu ile %90 barajını aşan ilk model olmuştur.
* **Teknik Yorum:** Ağın derinlik kapasitesi öznitelik çıkarmada çok başarılıdır ancak katman sayısının fazlalığı ve parametre yoğunluğu, bu modelin kenar cihazlarda (Edge AI), mikrodenetleyicilerde veya mobil platformlarda çalıştırılmasını zorlaştıracak bir hesaplama yükü doğurmaktadır.

### 3. MobileNetV2 Mimarisi
Mobil ve gömülü sistemler için optimize edilmiş, geleneksel konvolüsyon yerine "Derinlemesine Ayrılabilir Konvolüsyon" (Depthwise Separable Convolution) kullanan, parametre dostu ve ters çevrilmiş artık blok (inverted residual) yapısına sahip bir mimaridir.

* **Sıfırdan Eğitim (Scratch):** Modelin dar ve derin tasarımı, PlantVillage veri setini sıfırdan beslediğimizde ağır bir gradyan kararsızlığına yol açmış ve ilk 4 epoch boyunca rastgele tahmin çizgisinde (`%10` doğrulukta) kilitli kalmıştır. Neticede yalnızca `%38.14` validation başarısı yakalayabilmiştir.
* **Transfer Öğrenme (ImageNet):** Ağırlıklar hazır geldiğinde model olağanüstü bir sıçrama yapmış, epoch süresini `~1.6 dakikaya (99 saniye)` indirerek **projenin en hızlı eğitime sahip ağı** olmuştur. Validation doğruluğu ise `%87.40` gibi oldukça rekabetçi bir seviyeye ulaşmıştır.
* **Teknik Yorum:** MobileNetV2, eğitim ve doğrulama kayıplarının kafa kafaya gitmesiyle projenin en kararlı ve varyansı en düşük modelidir. Tepe noktadaki doğruluğu EfficientNetB0'ın bir tık gerisinde kalsa da donanım/performans verimliliğinde liderdir.

### 4. EfficientNetB0 Mimarisi
Ağ genişliğini, derinliğini ve giriş çözünürlüğünü birleşik bir katsayıyla (Compound Scaling) dengeli şekilde ölçekleyen, çağdaş ve son derece efektif bir derin öğrenme mimarisidir.

* **Sıfırdan Eğitim (Scratch):** Gelişmiş yapısına rağmen sıfırdan eğitimde zayıf öznitelik üretmiş, kararsız ve çok dalgalı bir loss grafiği çizerek erken aşamada ezberleme evresine geçmiştir (`%49.02` validation doğruluğu, `~8.6 dakika` epoch süresi).
* **Transfer Öğrenme (ImageNet):** ImageNet desteğiyle model **%91.81 validation doğruluğuna** ulaşarak projenin mutlak şampiyonu olmuştur. Üstelik `0.2475` gibi projenin en düşük hata (loss) oranını sıfır ezberleme (overfitting) ve `~2.2 dakika` gibi muazzam bir epoch süresi eşliğinde sunmuştur.
* **Teknik Yorum:** EfficientNetB0, matematiksel optimizasyonun doruk noktasıdır. Parametreleri bütçeli kullanırken doğruluğu maksimuma çıkarma yeteneği, onu büyük ölçekli ve yüksek doğruluk gerektiren sunucu sistemleri için rakipsiz kılmaktadır.

---

## 📉 Diğer Modelleri Eleme Nedenlerimiz (Mühendislik Kararları)

1.  **VGG16 Neden Elendi?:** Günümüz standartlarına göre aşırı yüksek parametre yüküne sahiptir. Transfer öğrenmede dahi MobileNetV2'den çok daha yavaş kalması ve buna rağmen daha düşük doğruluk üretmesi mimariyi tamamen verimsiz kılmaktadır.
2.  **ResNet-50 Neden Elendi?:** `%90.09` gibi başarılı bir doğruluğa ulaşmış olsa da, EfficientNetB0 mimarisinin hem çok daha yüksek doğruluk (`%91.81`) sunması hem de ResNet-50'ye kıyasla **yarı yarıya daha hızlı eğitilmesi (3.9 dk vs 2.2 dk)** ResNet-50'nin rasyonelliğini kaybettirmiştir.

---

## 🎯 Dağıtım (Deployment) Stratejisi ve Sonuç

Elde edilen benchmark sonuçlarına göre gerçek hayat senaryolarında iki farklı canlıya alım (deployment) stratejisi önerilmektedir:

* **Edge AI & Mobil Entegrasyon (Drone, Otonom Robot, Akıllı Telefon):** Saniyedeki kare sayısı (FPS), düşük RAM tüketimi ve `99 saniyelik` rekor epoch hızı göz önüne alındığında, tarla üzerinde anlık tarama yapacak drone veya mobil uygulamalarda **MobileNetV2** konuşlandırılmalıdır.
* **Bulut Tabanlı & Merkezi Analiz (Sunucu Sistemleri):** Çiftçilerin fotoğrafları bir merkeze yüklediği, hızdan ziyade milimetrik teşhisin ve en düşük hata payının hedeflendiği sunucu senaryolarında, projenin şampiyonu olan **EfficientNetB0** mimarisi kullanılmalıdır.
