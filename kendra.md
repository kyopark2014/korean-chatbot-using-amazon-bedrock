# Kendra 를 이용한 RAG의 구현

## 정확도 개선 방안

### Kendra API
Kendra에서 검색할때에 사용하는 API에는 [Retrieve API](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)와 [Query](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_Query.html)가 있습니다. 아래와 같이 요약하여 설명합니다. 

- Retrieve API는 Query API보다 많은 token으로 구성된 발췌문을 제공하는데, 발췌문의 길이는 RAG의 정확도에 매우 중요한 요소입니다. 또한 Retrieve API에 대한 token 숫자는 기본이 300인데, case를 통해 증량을 요청할 수 있습니다.
- Query API로 한글문서를 검색하는 경우에 token숫자의 제한으로 많은 경우에 만족할만한 RAG의 결과를 얻을 수 없었습니다.
- 검색의 정확도(score)를 활용하여 검색의 범위를 조정하면 RAG의 정확도가 올라갑니다. 그런데, Retrieve는 2023년 11월(현재)까지 영어(en)에 대해서만 score를 제공하고 있습니다. 따라서, 한국어(ko)는 token수가 적은 Query API를 이용할때만 score를 활용할 수 있습니다.
- Kendra의 [FAQ를 이용](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra-faq.md)하면 RAG의 정확도를 개선할 수 있는데, Query API로만 결과를 얻을 수 있습니다. 또한, Kendra에서는 Retrieve API로 조회시 결과가 없을때에 Query API로 fallback을 best practice로 가이드하고 있습니다. 따라서, FAQ를 사용하고자 한다면, Retrive와 Query API를 모두 사용하여야 합니다.

### LangChain 활용 방법

- LangChain은 Retrieve API로 검색하였을대에 결과가 없으면, Query로 한번 더 검색을 수행합니다.
- Kendra에서 한국어 문서를 업로드하면 retriever의 Language 설정을 "ko"로 설정하여야 합니다.
    
### 정확도 개선 방안

- 한글문서의 언어설정을 "ko"로하여 Kendra에 등록합니다.
- LangChain의 Kendra Retriever로 질의시 language를 "ko"로 설정하여야 retriever api로 더 많은 token을 가지는 발췌문을 얻을 수 있습니다.
- Kendra의 Retrieve/Query API로 직접 조회하면 좀더 유연하게 RAG를 구현할 수 있습니다.
- FAQ문서가 있다면, Kendra에 등록하여 활용합니다. FAQ 사용시 Query API를 활용하여하므로, 결과를 얻는 속도를 개선하기 위해 동시에 Retrieve/Query API를 호출합니다.


## API 가이드

### Retrieve API

[Retrieve](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)는 Default Quota 기준으로 하나의 발췌문(passges)는 200개의 token으로 구성될 수 있고, 최대 100개(PageSize)까지 이런 발췌문을 얻을 수 있습니다. 200 개의 token으로 구성된 발췌문(passage)과 최대 100개의 의미론적으로 관련된 발췌문을 검색할 수 있습니다. Query API와 다르게 qustion/answer와 FAG는 포함되지 않습니다. 

Retrieve API는 영어(en)만 score를 제공하고, 성능을 개선하기 위한 feedback을 지원하지 않습니다.

### Query API

[Query](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_Query.html)의 결과는 "DOCUMENT", "QUESTION_ANSWER", "ANSWER"의 Type이 있습니다. 

- ANSWER: 관련 제안된 답변(Relevant suggested answers)으로 text나 table의 발취(excerpt)로서 강조 표시(highlight)를 지원합니다. 
- QUESTION_ANSWER: 관련된 FAQ(Matching FAQs) 또는 FAQ 파일에서 얻은 question-answer입니다.
- DOCUMENT: 관련된 문서(Relevant documents)로서 문서의 발취(excerpt)와 title을 포하한 결고로서 강조 표시(hightlight)를 지원합니다.

관련 파라메터는 아래와 같습니다.

- QueryResultTypeFilter로 type을 지정할 수 있습니다.
- PageSize: 관련된 문장을 몇개까지 가져올지 지정합니다.
- PageNumber: 기본값은 결과의 첫페이지입니다. 첫페이지 이후의 결과를 가져올때 지정합니다.

결과를 가져오기

```python
def get_retrieve_using_Kendra(index_id, query, top_k):
    kendra_client = boto3.client(
        service_name='kendra', 
        region_name=kendra_region,
        config = Config(
            retries=dict(
                max_attempts=10
            )
        )
    )

    attributeFilter = {
        "AndAllFilters": [
            {
                "EqualsTo": {
                    "Key": '_language_code',
                    "Value": {
                        "StringValue": 'en',
                    },
                },
            },
        ],
    }

    try:
        resp =  kendra_client.query(
            IndexId = index_id,
            QueryText = query,
            PageSize = top_k,
            #PageNumber = page_number,
            #AttributeFilter = attributeFilter,
            #QueryResultTypeFilter = "DOCUMENT",  # 'QUESTION_ANSWER'
        )
    except Exception as ex:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
        
        raise Exception ("Not able to retrieve to Kendra")        
    print('resp, ', resp)
    print('resp[ResultItems], ', resp['ResultItems'])
    
    for query_result in resp["ResultItems"]:
        print("-------------------")
        print("Type: " + str(query_result["Type"]))
            
        if query_result["Type"]=="ANSWER" or query_result["Type"]=="QUESTION_ANSWER":
            answer_text = query_result["DocumentExcerpt"]["Text"]
            print(answer_text)
    
        if query_result["Type"]=="DOCUMENT":
            if "DocumentTitle" in query_result:
                document_title = query_result["DocumentTitle"]["Text"]
                print("Title: " + document_title)
            document_text = query_result["DocumentExcerpt"]["Text"]
            print(document_text)
    
        print("------------------\n\n")      
```



## Score

[ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)와 같이 "VERY_HIGH", "HIGH", "MEDIUM", "LOW", "NOT_AVAILABLE"로 결과의 신뢰도를 확인할 수 있습니다.


## FAQ (Frequently asked questions)

[FAQ-Kendra](https://github.com/aws-samples/enterprise-search-with-amazon-kendra-workshop/blob/master/Part%202%20-%20Adding%20a%20FAQ.md)를 참조합니다. [kendra-faq-refresher](https://github.com/aws-samples/amazon-kendra-faq-refresher/tree/main)를 참조하여 FAQ를 Kendra 검색 결과로 활용할 수 있습니다.

여기서는 [Kendra FAQ](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra-faq.md)와 같이 Kendra의 Query API로 FAQ의 결과를 얻으면, 질문과 결과를 함께 excerpt로 활용합니다.

## Kendra 성능향상 방법

[Relevance tuning with Amazon Kendra](https://aws.amazon.com/ko/blogs/machine-learning/relevance-tuning-with-amazon-kendra/)

[Submitting feedback for incremental learning](https://docs.aws.amazon.com/kendra/latest/dg/submitting-feedback.html)

[Tuning search relevance](https://docs.aws.amazon.com/kendra/latest/dg/tuning.html)

## User/Group별 활용

Document에 대한 접근권한을 관리할 수 있도록 [UserContext](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_UserContext.html)를 이용합니다. 이때, [DataSourceGroup](https://docs.aws.amazon.com/ko_kr/kendra/latest/APIReference/API_DataSourceGroup.html)는 데이터 소스 그룹별로 검색을 아래와 같이 지정할 수 있습니다.

```
response = client.query(
    IndexId='string',
    QueryText='string',
    UserContext={
        'Token': 'string',
        'UserId': 'string',
        'Groups': [
            'string',
        ],
        'DataSourceGroups': [
            {
                'GroupId': 'string',
                'DataSourceId': 'string'
            },
        ]
    },
```

## Reference

[Retrieve API](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)

[Enterprise Search with Amazon Kendra](https://github.com/aws-samples/enterprise-search-with-amazon-kendra-workshop/tree/master)


## Retrieve 예제

```java
{
   "QueryId":"4dcde662-831f-438d-8870-98d4fa27ded2",
   "ResultItems":[
      {
         "Id":"4dcde662-831f-438d-8870-98d4fa27ded2-acd1426d-9c78-49d8-9bab-fa826d443413",
         "DocumentId":"0bc8608c-aff6-4171-8ca2-0634d466adaf",
         "DocumentTitle":"fsi_faq_ko.csv",
         "Content":"Category\tInformation\ttype\tSource 아마존 은행의 타기관OTP 이용등록방법 알려주세요\t아마존 은행의 타기관에서 발급받으신 OTP가 통합OTP카드인 경우 당행에 등록하여 이용가능합니다. [경로] - 인터넷뱅킹 로그인→ 사용자관리→인터넷뱅킹관리→OTP이용등록 - 아마존은행 모바일앱 로그인→ 전체메뉴→설정/인증→ 이용중인 보안매체선택→   OTP이용등록 ※ OTP이용등록후 재로그인을 하셔야 새로 등록된 보안매체가 적용됩니다. 기타 궁금하신 내용은 아마존 은행 고객센터 1599-9999로 문의하여 주시기 바랍니다. 인터넷뱅킹\t아마존은행 아마존 공동인증서와 금융인증서 차이점이 무엇인가요? 공동인증서 (구 공인인증서)는 용도에 따라 은행/신용카드/보험용 무료 인증서와 전자거래범용(수수료 4,400원) 인증서가 있으며 유효기간은 1년입니다. ※ OTP이용등록후 재로그인을 하셔야 새로 등록된 보안매체가 적용됩니다. 기타 궁금하신 내용은 아마존 은행 고객센터 1599-9999로 문의하여 주시기 바랍니다. 인터넷뱅킹\t아마존은행 아마존 공동인증서와 금융인증서 차이점이 무엇인가요? 공동인증서 (구 공인인증서)는 용도에 따라 은행/신용카드/보험용 무료 인증서와 전자거래범용(수수료 4,400원) 인증서가 있으며 유효기간은 1년입니다. 아마존 공동인증서는 하드디스크나 이동식디스크, 휴대폰 등 원하시는 기기에 저장해서 이용할 수 있습니다. 인증서를 저장한 매체에서는 인증서 비밀번호로 편리하게 이용할 수 있으나 다른 기기에서 이용하려면 기기마다 복사하거나 이동식디스크에 저장해서 휴대해야 하는 불편함이 있을 수 있습니다. 아마존 금융인증서는 금융인증서는 금융결제원의 클라우드에 저장하여 이용하는 인증서로 발급/이용 시에 클라우드에 접속이 필요합니다. 금융결제원 클라우드에 연결만 가능하다면 어디서든 편리하게 이용 가능하지만, PC나 USB, 휴대폰 등 다른 기기로 복사는 불가합니다. (유효기간 3년/발급 수수료 무료) ※ 클라우드 계정 연결을 위해 휴대폰을 통한 ARS, SMS, 마이인포앱 인증 절차가 필요합니다. 인증서\t아마존은행 공동인증서와 금융인증서 차이점이 무엇인가요? 공동인증서 (구 공인인증서)는 용도에 따라 은행/신용카드/보험용 무료 인증서와 전자거래범용(수수료 4,400원) 인증서가 있으며 유효기간은 1년입니다. 공동인증서는 하드디스크나 이동식디스크, 휴대폰 등 원하시는 기기에 저장해서 이용할 수 있습니다. 인증서를 저장한 매체에서는 인증서 비밀번호로 편리하게 이용할 수 있으나 다른 기기에서 이용하려면 기기마다 복사하거나 이동식디스크에 저장해서 휴대해야 하는 불편함이 있을 수 있습니다. 금융인증서는 금융인증서는 금융결제원의 클라우드에 저장하여 이용하는 인증서로 발급/이용 시에 클라우드에 접속이 필요합니다. 금융결제원 클라우드에 연결만 가능하다면 어디서든 편리하게 이용 가능하지만, PC나 USB, 휴대폰 등 다른 기기로 복사는 불가합니다. (유효기간 3년/발급 수수료 무료) ※ 클라우드 계정 연결을 위해 휴대폰을 통한 ARS, SMS, 마이인포앱 인증 절차가 필요합니다. 인증서\t서울은행 타기관OTP 이용등록방법 알려주세요\t타기관에서 발급받으신 OTP가 통합OTP카드인 경우 당행에 등록하여 이용가능합니다. [경로] - 인터넷뱅킹 로그인→ 사용자관리→인터넷뱅킹관리→OTP이용등록 - 서울 모바일앱 로그인→ 전체메뉴→설정/인증→ 이용중인 보안매체선택→   OTP이용등록 ※ OTP이용등록후 재로그인을 하셔야 새로 등록된 보안매체가 적용됩니다. 기타 궁금하신 내용은 서울은행 고객센터 1599-8000로 문의하여 주시기 바랍니다. 인터넷뱅킹\t서울은행 공동인증서와 금융인증서 차이점이 무엇인가요?",
         "DocumentURI":"https://d2me0ac2n5hgqe.cloudfront.net/docs/fsi_faq_ko.csv",
         "DocumentAttributes":[
            {
               "Key":"_source_uri",
               "Value":{
                  "StringValue":"https://d2me0ac2n5hgqe.cloudfront.net/docs/fsi_faq_ko.csv"
               }
            }
         ],
         "ScoreAttributes":{
            "ScoreConfidence":"NOT_AVAILABLE"
         }
      },      
   ],
   "ResponseMetadata":{
      "RequestId":"5dd6b3ea-1774-4b6c-b88e-c1cfbfeb0e9a",
      "HTTPStatusCode":200,
      "HTTPHeaders":{
         "x-amzn-requestid":"5dd6b3ea-1774-4b6c-b88e-c1cfbfeb0e9a",
         "content-type":"application/x-amz-json-1.1",
         "content-length":"45160",
         "date":"Mon, 20 Nov 2023 00:34:25 GMT"
      },
      "RetryAttempts":0
   }
}
````

### QUESTION_ANSWER

```java
[
   {
      "Id":"74719041-8126-473c-92f1-929fdc520138-188b319d-552f-4ff4-a7d5-8cbcd21dbea8",
      "Type":"QUESTION_ANSWER",
      "Format":"TEXT",
      "AdditionalAttributes":[
         {
            "Key":"QuestionText",
            "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
            "Value":{
               "TextWithHighlightsValue":{
                  "Text":"How many free clinics are in Spokane WA?",
                  "Highlights":[
                     {
                        "BeginOffset":4,
                        "EndOffset":8,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":9,
                        "EndOffset":13,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":14,
                        "EndOffset":21,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":29,
                        "EndOffset":36,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":37,
                        "EndOffset":39,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     }
                  ]
               }
            }
         },
         {
            "Key":"AnswerText",
            "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
            "Value":{
               "TextWithHighlightsValue":{
                  "Text":"13",
                  "Highlights":[
                     
                  ]
               }
            }
         }
      ],
      "DocumentId":"c24c0fe9cbdfa412ac58d1b5fc07dfd4afd21cbd0f71df499f305296d985a8c9a91f1b2c-e28b-4d13-8b01-8a33be5fc126",
      "DocumentTitle":{
         "Text":""
      },
      "DocumentExcerpt":{
         "Text":"13",
         "Highlights":[
            {
               "BeginOffset":0,
               "EndOffset":2,
               "TopAnswer":false,
               "Type":"STANDARD"
            }
         ]
      },
      "DocumentURI":"https://www.freeclinics.com/",
      "DocumentAttributes":[
         {
            "Key":"_source_uri",
            "Value":{
               "StringValue":"https://www.freeclinics.com/"
            }
         }
      ],
      "ScoreAttributes":{
         "ScoreConfidence":"VERY_HIGH"
      },
      "FeedbackToken":"AYADeN-jZ9DvGVP9n00b4d48LrsAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREF6ajZZVkJ3M3B4dXZEMGRJZitQaEEzUWNVZkE3TDVBbjNEOCs1bE1aRm1hN1M3a0N3cjNiMzZRR2hPcTloeVJ1QT09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwEP6KAczOsL3xpUp6oizSAgAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMT_QgV_BMu5l49EZAAgEQgDuTOkP1QJbt85KZ4FDF438i0upluDZq_Rf3L8H9PqkLQOSgUAyyy9hqEmMOZUcGqvBNc_ekw4pbMRy5ZAIAAAAADAAAEAAAAAAAAAAAAAAAAAB3LYEFCQuAEb8NnKHSk1eT_____wAAAAEAAAAAAAAAAAAAAAEAAAF2QLnDNTO_Ma1EGreEOHC8YG5ijJ7jLblLE4CbyAY9ueJHKBTQ-Rf2A_pD9hpXTuyP6Ho84IIlScm7IhFUomBUSZMD_qrc0qnvlrCjgXwJ_AM0MJKmqBkMPNvivFnfZ9xl-dFyFX1sdzq0_LUE4KgLZpjQiSU0b_PFJw2zN8P6JSJb9Fz84fbWu1_nzrVrqCj5dDpMLDNLgC3f6pTS4IqmJqMsj6BbGcdsvLIzVA2XaAGYS8CNv9pu5Hz63yrh6hG4UHWJwdhIcPZG7z7BayFjravsKjw101PJnzUKSIfiZlRnoqm-Bbff-ieECV-vZ_1vtskbHhmsZ4WlKTcpD5QGMrElbk7WMbdPf8gmGQfC8SMrR-ixO7d856LIsoTx9i6VcN91GxEKcYtsXY4J0w6G4aL8-tj1iS7zwIsxHimIsuAHM4u5SmHmI_oJ25pR-7TA2K34GVv9VhYydG8JsBbjGV-mPpg6ORE4bNkhRL38f1pfHEXNlv79F9b8UP93MLfj6lZT25tPAGcwZQIxAK_GncCyOyt2NLdszY-Oc2Qchpo2CCTjj25a5wyzYv4JObw591oaxZeSVbA_Mq2v2gIwf6xq0c5vYGait9J9mnI2FMtEJ3rI2DRld30IRmWWNES54XOxciMd5J_YxJGkfghX.74719041-8126-473c-92f1-929fdc520138-188b319d-552f-4ff4-a7d5-8cbcd21dbea8"
   }
]
```


### Document

```text
{
   "QueryId":"cab6a783-5daf-40b6-a3d9-4df51f3ec812",
   "ResultItems":[
      {
         "Id":"0732b19c-333b-4f60-9724-323b54928f52-3f4bf464-5098-4463-b14d-eee05828fe0e",
         "Type":"DOCUMENT",
         "Format":"TEXT",
         "AdditionalAttributes":[
            
         ],
         "DocumentId":"e3f2af36-f76b-4775-a5ed-863910bc7a64",
         "DocumentTitle":{
            "Text":"book_smmdinlm239.pdf",
            "Highlights":[
               
            ]
         },
         "DocumentExcerpt":{
            "Text":"...있으며, 가입자는 가입에 앞서 이에 대한 충분한 설명을 받으시기 바랍니다.\n\n\n•가입할 때 보험계약의 기본사항[보험상품명, 보험기간, 납입기간, 피보험자 등]을 반드시 확인하시기 바랍니다.\n\n\n• 보험계약 청약서에서 질문한 사항(계약 전 알릴 의무 사항)에 대해 고의 또는 중대한 과실로 사실과 다르게 알린 경우에는  \n\n\n보험계약이 해지되거나 보장이 제한될 수 있습니다...",
            "Highlights":[
               
            ]
         },
         "DocumentURI":"",
         "DocumentAttributes":[
            {
               "Key":"_excerpt_page_number",
               "Value":{
                  "LongValue":5
               }
            }
         ],
         "ScoreAttributes":{
            "ScoreConfidence":"MEDIUM"
         },
         "FeedbackToken":"AYADeOHYuP8KBBZVDtqxvB8JoHoAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREF6K2JUSmJlM0hHU2Y2SktqdXhwaDNsOFEyTWdtYjRzQ2UvTElvdDZ3dElaczFDV21sc1dZYzhhSDdwaitNQmF0dz09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwHTuiBoobRJuvQ0u6-6abBiAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMYJMTvVwth6_ptw6pAgEQgDsrzR_f7IbDgqT5XrW3v5Da-aTNFw2v7fpcIra8PD02pQ5feUAw_JkXR81MXrqrbQ7fKsEexQPdTsx3JAIAAAAADAAAEAAAAAAAAAAAAAAAAACbC35XEBh-MXfq3LU5gDvR_____wAAAAEAAAAAAAAAAAAAAAEAAAF2PUZ0ZVJdz40rYWthpb1aYMLWCD1h3sVB6aXfhcUKwnKyilNtQ6Sr-yq_B0Daajd5P7PEwDeuPRmwAh164ukTXwP9qJIiaADhiCYTD0BZrtCC_JdH9wudYz0ximAEWA4pi1A0NSucA6RxmJRK16VXfYLxp7JIooWL4SLyoCxmRIM1ZwNs38YmnwEh5XA-crpcaQBTBgjkVejsmCCbNh1RVylDpmkMUxzYZZh2uIOYFAmjocr9IbFJKQz1IQNktcq-_uUdaO6jvNNUTUsCoJMMM4A4H4yczVD3RpJGNzatYHYRq8zhhmF6B9PGaVbDD26FrH9uyryyY9XhdL4GMBm1C4DPCDmWBEqfD9dCgfn32784tSJklA8xp9C_46hq_UyYy8uSWjbmJYNu0rRoueGJrkmcRV1sA9gUDbriYr4Ij249GahhRsF8nsDWtkmOzZbBU0CXsmklH4A7K7MBnBBsEcFDQE9Wxi-VNPHcgjmAL-ilMHNY6B0IYiDB_w9lNp6eebc8LZ4bAGcwZQIxAPRf5aJNC67fcRlu61G66HRAQ6m-rvg3eWw4GqrZ4WvtBuZ_GTxDvve-CYWpGtedpwIwLJHMr9WZLpByNqsyLhQk3ztG4eIA830nRg4rz9Odzp9mVlD2-bh8aOHx-r0JXz23.0732b19c-333b-4f60-9724-323b54928f52-3f4bf464-5098-4463-b14d-eee05828fe0e"
      }
   ],
   "FacetResults":[
      
   ],
   "TotalNumberOfResults":4,
   "ResponseMetadata":{
      "RequestId":"52b5d039-a600-4ed8-90b1-d1fa46c5e1c7",
      "HTTPStatusCode":200,
      "HTTPHeaders":{
         "x-amzn-requestid":"52b5d039-a600-4ed8-90b1-d1fa46c5e1c7",
         "content-type":"application/x-amz-json-1.1",
         "content-length":"9569",
         "date":"Fri, 17 Nov 2023 06:55:10 GMT"
      },
      "RetryAttempts":0
   }
}
```
