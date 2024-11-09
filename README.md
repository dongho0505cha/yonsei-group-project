# yonsei-group-project

## Fact Check 동작 원리
0. answer 생성
1. answer 를 paragraph(라인) 단위로 쪼갬
2-1. paragraph에서 initial 찾기
2-1. NLTK(Natural Language Toolkit)의 sent_tokenize 를 사용하여 긴 텍스트(paragraph)를 개별 문장 단위로 나눔
=> 하나의 paragraph 안의 여러 문장을 쪼개는 역할
3. 분할된 문장 재구성(이니셜로 잘못 분할된 문장 병합, 너무 짧은 문장을 다른 문장과 결합 등)
4. 문장과 샘플을 이용하여 프롬프트 작성
5. LLM에 문장을 atomic 단위로 분할 요청
6. fact 유형이 아닌 문장을 제외하여 sentence-atomic pair를 구성
7. paragraph에 따른 sentence-atomic pair 후처리 작업
8. knowledge_source 와 atomic 으로 LLM을 통해 fact 유무를 확인
9. atomic들의 fact 유무를 종합하여 평균 값을 구함
