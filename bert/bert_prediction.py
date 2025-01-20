import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import emoji
import re
from zeyrek import MorphAnalyzer
from nltk.corpus import stopwords

class OccupationPredictor:
    def __init__(self, model_path="./saved_model"):
        # Model ve tokenizer'ı yükle
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Label dictionary'yi yükle
        with open(f'{model_path}/label_dict.json', 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)
            
        # Label dictionary'yi ters çevir (index -> meslek)
        self.reverse_label_dict = {int(v): k for k, v in self.label_dict.items()}
        
        # Model'i evaluation moduna al
        self.model.eval()
        # Türkçe stop words'ları hazırla
    def get_stop_words(self):
        stop_words = set(stopwords.words('turkish'))
        
        # Ek stop words eklenebilir
        extra_stop_words = {
            'rt', 'via', 'http', 'https', 'www', 
            'com', 'org', 'html', 'amp', 'retweet',
            've', 'ile', 'da', 'de', 'mi', 'mu', 'mü', 'mı'
        }
        stop_words.update(extra_stop_words)
        
        return stop_words


    def clean_text(self,text):

        if text is None:
            return ''
        # Emojileri temizle
        text = emoji.replace_emoji(text, replace='')
        
        # URL'leri temizle
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Kullanıcı adlarını temizle
        text = re.sub(r'@\w+', '', text)
        
        # Hashtag'leri temizle
        text = re.sub(r'#\w+', '', text)
        
        # Noktalama işaretlerini temizle
        text = re.sub(r'[^\w\s]', '', text)

        # Kaçış karakterlerini temizle
        text = re.sub(r'\\n|\\t|\\r|\\', ' ', text)
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # Kelimenin kökünü bulma fonksiyonu
    def find_root(self,token, analyzer):
        lemmatized = analyzer.lemmatize(token)
        return lemmatized

    # Ön işleme fonksiyonu
    def preprocess(self,text, stop_words, analyzer):
        # Metni temizle
        cleaned_text = self.clean_text(text)
        
        # Tokenize et (kelimelere ayır)
        tokens = cleaned_text.split()
        
        # Stop words'ları çıkar
        tokens_without_stopwords = [
            token for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        
        # Köklere ayır
        root_tokens = [self.find_root(token, analyzer) for token in tokens_without_stopwords]

        lemmatized_tokens = []
        for token in root_tokens:
            lemmatized_tokens.append(token[0][1][0])
        
        lemmatized_tweet = ' '.join(lemmatized_tokens)

        return lemmatized_tweet
    def predict(self, text):

        # Stop words'ları yükle
        stop_words = self.get_stop_words()

        # Zeyrek morfoloji analizörünü yükle
        analyzer = MorphAnalyzer()

        # Metni ön işle
        processed_text = self.preprocess(text, stop_words, analyzer)
        
        # Metni tokenize et
        inputs = self.tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Tahmin yap
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        # Tahmin olasılıklarını al
        probabilities = predictions[0].tolist()
        
        # En yüksek olasılıklı 3 mesleği ve olasılıklarını bul
        top_3_indices = sorted(range(len(probabilities)), 
                             key=lambda i: probabilities[i], 
                             reverse=True)[:3]
        
        top_3_predictions = [
            {
                'meslek': self.reverse_label_dict[idx],
                'olasilik': round(probabilities[idx] * 100, 2)
            }
            for idx in top_3_indices
        ]
        
        return {
            'orijinal_tweet': text,
            'islenmis_tweet': processed_text,
            'tahmin_edilen_meslek': self.reverse_label_dict[predicted_class],
            'en_yuksek_olasilikli_3_meslek': top_3_predictions
        }

def main():
    # Tahmin yapıcıyı başlat
    predictor = OccupationPredictor()
    
    # Örnek kullanım
    while True:
        # Kullanıcıdan tweet al
        print("\nTahmin edilecek tweet'i girin (Çıkmak için 'q' yazın):")
        text = input()
        
        if text.lower() == 'q':
            break
        
        # Tahmin yap
        result = predictor.predict(text)
        
        # Sonuçları göster
        print("\nTahmin Sonuçları:")
        print(f"\nOrijinal Tweet: {result['orijinal_tweet']}")
        print(f"İşlenmiş Tweet: {result['islenmis_tweet']}")
        print(f"\nTahmin edilen meslek: {result['tahmin_edilen_meslek']}")
        print("\nEn yüksek olasılıklı 3 meslek:")
        for pred in result['en_yuksek_olasilikli_3_meslek']:
            print(f"- {pred['meslek']}: %{pred['olasilik']}")

if __name__ == "__main__":
    main()