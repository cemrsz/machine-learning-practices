Machine Learning Practices

Bu repo, makine öğrenmesi öğrenme sürecimde hazırladığım küçük, bağımsız ve uygulama odaklı çalışmaları içerir. Her klasör veya dosya; bir veri kümesini inceleme, modeli eğitme, sonucu değerlendirme ya da görselleştirme gibi belirli bir kavramı pratik etmeye odaklanır.

Amaç yalnızca hazır bir model çalıştırmak değil; veriyi tanımak, doğru özellikleri seçmek, modelin kararını yorumlamak ve farklı yaklaşımları karşılaştırarak makine öğrenmesi temellerini pekiştirmektir.

## İçerik

Repo zamanla yeni çalışmalarla genişler. Örnek konu başlıkları:

- Veri okuma, temizleme ve keşifsel veri analizi
- Eğitim/test verisi ile çalışma
- Sınıflandırma ve regresyon problemleri
- Karar ağaçları ve Gini safsızlığı
- Model doğrulama ve çapraz doğrulama
- Matplotlib ve Seaborn ile veri/model görselleştirme
- Özellik seçimi, hiperparametre denemeleri ve sonuç yorumlama

## Örnek çalışma: Karar ağacı

Repo içindeki karar ağacı pratiği, seçim verisinden ikili bir hedef değişken oluşturarak sınıflandırma yapar. Çalışmada:

- CSV verisi Pandas ile okunur,
- hedef değişken `0` ve `1` olarak hazırlanır,
- değişkenlerin sınıflara göre dağılımı görselleştirilir,
- `DecisionTreeClassifier` ile model eğitilir,
- oluşan ağaç görsel olarak incelenir.

Bu örnek, karar ağacının veriyi hangi eşiklerde böldüğünü ve sınıflandırma mantığını anlaşılır hale getirmek için hazırlanmıştır.

## Gereksinimler

- Python 3.9 veya üstü
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

Kurulum:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Veri kümeleri

Her çalışma, kendi klasöründeki yönergelere ve veri dosyalarına ihtiyaç duyabilir. Karar ağacı örneği için aşağıdaki dosyaları `data/` klasörüne ekleyin:

```text
data/county_election_train.csv
data/county_election_test.csv
```

Bu örnekte CSV dosyalarında en az şu sütunların bulunması gerekir:

| Sütun | Açıklama |
| --- | --- |
| `trump` | Trump'ın aldığı oy/değer |
| `clinton` | Clinton'ın aldığı oy/değer |
| `minority` | Azınlık nüfusu ile ilgili sayısal özellik |
| `bachelor` | Lisans mezuniyeti ile ilgili sayısal özellik |

## Çalıştırma

Önce gerekli paketleri yükleyin. Ardından çalıştırmak istediğiniz Python dosyasını proje kökünden çalıştırın:

```bash
python "Visualizing decision tree.py"
```

Kod iki görsel üretir:

1. `minority` ve `bachelor` değerlerinin Trump/Clinton sınıfına göre dağılımı
2. Eğitim verisi üzerinden oluşturulan karar ağacı

## Öğrenme yaklaşımı

Her pratikte genel olarak şu akış takip edilir:

1. Veriyi oku ve temel yapısını incele.
2. Eksik, hatalı veya gereksiz bilgileri değerlendir.
3. Hedef değişkeni ve özellikleri belirle.
4. Uygun modeli eğit.
5. Sonucu doğruluk, çapraz doğrulama veya görsellerle değerlendir.
6. Modelin güçlü ve sınırlı yönlerini yorumla.

Karar ağacı örneğinde hedef değişken şu kuralla üretilir:

```python
y_train = np.where(elect_train["trump"] > elect_train["clinton"], 1, 0)
```

Ardından model yalnızca `minority` değişkeniyle eğitilir:

```python
model = DecisionTreeClassifier(max_depth=3, criterion="gini")
model.fit(elect_train[["minority"]], y_train)
```

- `max_depth=3`: Ağacın aşırı karmaşıklaşmasını sınırlar.
- `criterion="gini"`: Her düğümde sınıfları en iyi ayıran bölünmeyi seçer.

## Geliştirme fikirleri

- Aynı veri kümesinde birden fazla modeli karşılaştırın.
- Model sonuçlarını sadece doğrulukla değil; precision, recall ve F1-score ile de inceleyin.
- Eğitim/test farkını gözlemleyerek overfitting durumlarını araştırın.
- `max_depth`, `min_samples_split` ve `min_samples_leaf` gibi hiperparametreleri deneyin.
- Kendi notlarınızı ve bulgularınızı her pratiğin yanına ekleyin.

## Not

Bu repo eğitim ve pratik amaçlıdır. Çalışmalar, kavramları uygulayarak öğrenmek için tasarlanmıştır; gerçek dünyada kullanılacak modeller için veri kalitesi, etik etkiler, yanlılık, güvenlik ve kapsamlı doğrulama ayrıca değerlendirilmelidir.
