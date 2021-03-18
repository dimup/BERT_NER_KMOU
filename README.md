# NER_BERTandKoBERT
한국해양대학교 개체명 코퍼스를 활용한 BERT 기반, KoBERT 기반 개체명 인식 모델 구현 내용입니다. 보다 고성능 환경에서 구현하기 위해 Google Colab에서 작성하였습니다.

### **파일 설명**
+ **"BERT_NER_KMOU.ipynb"**
  + Google BERT 기반의 개체명 인식 모델을 구현하는 과정 및 코드가 작성된 파일입니다.
+ **"KoBERT_NER_KMOU.ipynb"**
  + SKTBrain KoBERT 기반의 개체명 인식 모델을 구현하는 과정 및 코드가 작성된 파일입니다.
  + BERT 기반 모델의 구현 과정과 동일한 흐름을 갖지만, Data Preprocessing, Modeling, Inference 에서 약간의 차이점이 있습니다.
+ **"NER_Data_Parsing_example.ipynb"**
  + 해양대학교 개체명 코퍼스에서 필요 부분만 파싱해오는 코드 예시입니다.


### **구현 과정**
**1. Data 수집 및 구축**
+ 한국해양대학교 개체명 코퍼스에서 input data와 target data 각각 약 21000 문장을 파싱한 뒤 Training, Validation 데이터로 분리합니다.
  + Training set : 약 17000 문장
  + Validation set : 약 4000 문장

**2. Input data, Target data 전처리**
+ BERT 구조 형식에 맞게 데이터 전처리를 진행합니다. Input data와 Target data의 전처리는 차이가 있지만, 다음의 공통 과정을 거칩니다.
  + [CLS], [SEP] 토큰 부착
  + WordPiece(BERT) SentencePiece(KoBERT) tokenizing
  + Embedding
  + Padding
  
**3. Modeling**
+ 사전학습된 BERT, KoBERT 모델에 Token Classification Layer를 쌓은 형태로 모델링을 진행합니다.
  + BERT : Huggingface의 transformers 라이브러리에서 BertForTokenClassification 클래스를 활용했습니다. 사전학습 모델은 BERT-multilingual.
  + KoBERT : BERT와 동일하지만, 사전학습 모델을 KoBERT 모델로 진행합니다. 
    + (monologg님깨서 transformers 라이브러리에서 바로 KoBERT를 사용할 수 있도록 설정하신 것을 활용했습니다. https://github.com/monologg/KoBERT-Transformers)

**4. Training**
+ Optimizer와 Hyper parameters를 다음과 같이 설정한 뒤 학습을 진행합니다.
  + Optimizer : AdamW optimizer
  + Learning rate : 1e-5
  + Epsilon : 1e-8
  + Epochs : 5
  + Batch size : 8
+ 학습을 마친 뒤 Validation을 수행합니다. 토큰 간 정확도(Accuracy)를 측정했습니다.
  + BERT :  
  + KoBERT : 94.03966784156759 %

**5. Testing**
+ 임의의 문장을 학습이 완료된 모델에 통과시켜 그 결과를 확인합니다.

<br>

### **참고**
+ 한국해양대학교 개체명 코퍼스 : https://github.com/kmounlp/NER
+ Google BERT : https://github.com/google-research/bert
+ SK T-Brain KoBERT : https://github.com/SKTBrain/KoBERT
+ huggingface.co : https://huggingface.co/transformers/model_doc/bert.html
