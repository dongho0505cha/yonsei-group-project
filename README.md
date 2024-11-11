# AI 전문가 과정 4조 팀 프로젝트

---
### 초기화
1. python 3.10 or 3.11 버전 설치
2. pip install -r requirements.txt 로 패키지 설치 (가상환경 선택사항)
3. python -m spacy download en_core_web_sm 실행
4. uvicorn main:app --reload 로 애플리케이션 실행(or 디버깅 실행)
5. 127.0.0.1:8000/docs => swagger 접속

---
### 리소스 정보
질문 데이터셋 : https://huggingface.co/datasets/donghocha/first-chunk-questions-dataset
pinecone db 인덱스 : wikipedia-persons 

---
### Fact Check 동작 원리
1. answer 생성
2. answer 를 paragraph(라인) 단위로 쪼갬
3. paragraph에서 initial 찾기
4. NLTK(Natural Language Toolkit)의 sent_tokenize 를 사용하여 긴 텍스트(paragraph)를 개별 문장 단위로 나눔
=> 하나의 paragraph 안의 여러 문장을 쪼개는 역할
5. 분할된 문장 재구성(이니셜로 잘못 분할된 문장 병합, 너무 짧은 문장을 다른 문장과 결합 등)
6. 문장과 샘플을 이용하여 프롬프트 작성
7. LLM에 문장을 atomic 단위로 분할 요청
8. fact 유형이 아닌 문장을 제외하여 sentence-atomic pair를 구성
9. paragraph에 따른 sentence-atomic pair 후처리 작업
10. knowledge_source 와 atomic 으로 LLM을 통해 fact 유무를 확인
11. atomic들의 fact 유무를 종합하여 평균 값을 구함
