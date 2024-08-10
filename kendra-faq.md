# Kendra에서 FAQ (Frequently Asked Questions) 의 활용

FAQ에 있는 "How many free clinics are in Spokane WA?"의 Answer는 13이고 아래와 같은 reference를 가지고 있습니다.

![noname](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/996174e6-765b-4d2c-a5ea-7cfeb838a609)

아래는 "How many clinics are in Spokane WA?"으로 Kenra에 Query 했을때의 결과입니다. 여기서 FAQ의 "free"가 있고 없음에 따라 결과는 매우 다름에도 불구하고 아래와 같이, Kendra의 응답은 "ScoreConfidence"를 "VERY_HIGH"로 응답하고 있습니다.

따라서, 응답의 Type이 "QUESTION_ANSWER"인 경우에는 발췌를 할때에 "AdditionalAttributes"의 "QuestionText"을 같이 사용하여야 합니다. 즉 "How many free clinics are in Spokane WA? 13"으로 사용합니다.

```python
{
   "QueryId":"6ca1e9c4-9ce1-41f7-a527-2b6a536ad8b4",
   "ResultItems":[
      {
         "Id":"6ca1e9c4-9ce1-41f7-a527-2b6a536ad8b4-e1b952fa-7d3b-4290-ac5e-9f12b99db3af",
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
         "FeedbackToken":"AYADeLqxw-UVD3GzCPho81xtW6IAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREFrdE4vUzFJTWFmVGVjaFhUbHhlLzh4VXFvYXNablozR2RmeHVFb0JHN05ZYVVoNmZnMVRUMGdjQS9CTzdWWlNNQT09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwGkHCq8G_rQwpPcBduAGoQFAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMxTLvEAoHAoxFuktZAgEQgDsadJ5kv-qG2Clv0RRB7DSV8yK1Hv92wtCDYRDp0qswISUtpo6BlAvfaqAG3VD__jmy_wyv_iPuJpQn3wIAAAAADAAAEAAAAAAAAAAAAAAAAAAkeJmB-aXVfLQSLWk6xKfQ_____wAAAAEAAAAAAAAAAAAAAAEAAAF2YTptEGgTPKdRbRK31PKlnqy7fC4-exZfkWr1KhaYjExSOuoIxntlROmX8DaVLp7Iy-TaUrKg9-C0Iwj62FlMEDKbBxVdq7jI31uIZDP58Z17HUvl4acvRktyW_gaPMIDiVxo0QaAKvmP7qwq34-3Ti_ODBP3dTaufr7atTjsyBBJTQgJ4P3SSfMPniqdZOQTUnIb5PcbALwGVTT3FLxu9LxlpscfyoKvzGZSLAPDgmRWINEmPz9j9h-IzUATlJtqpydOnX3wAmUKx4GyBzISuhU7IxXK0BnYBAcwkl_ii04W5vEUi02cyLOBTU8iwYo-C0xUY7X2IccrzGHxF4o3YPW6mnmC_6YXm2nXPUi0OOPoX1sk5kuhx8Ra3n11DuKq6SAGH414hSxqFrPSfqBCIF5356z4tG_nJQ6KxKAi2LUwypAAme48DbZH_5KTsfFcm7X3MSr23XwNhq8jDbudlz-M3JF_6E314B7V99VASAxNxk-yiOUyjcMoECtcQ-cpgAbOf2MEAGcwZQIxAMtWD1KR4dhKMBJyJMOwoFsRgdVONBxgaz3z2TKJZfSUl3l-CFyENlB9oWq1f_dlOgIwXVTlmVXN5gyOZvvlqfvJybhNO2tyZ53ilC5_GsKl9Lu9cRnGjMX-tKqPyheSCVc3.6ca1e9c4-9ce1-41f7-a527-2b6a536ad8b4-e1b952fa-7d3b-4290-ac5e-9f12b99db3af"
      }
   ],
   "FacetResults":[
      
   ],
   "TotalNumberOfResults":1,
   "ResponseMetadata":{
      "RequestId":"f415076d-1384-47d4-8718-8245ca3c6d89",
      "HTTPStatusCode":200,
      "HTTPHeaders":{
         "x-amzn-requestid":"f415076d-1384-47d4-8718-8245ca3c6d89",
         "content-type":"application/x-amz-json-1.1",
         "content-length":"2642",
         "date":"Sat, 18 Nov 2023 03:24:27 GMT"
      },
      "RetryAttempts":0
   }
}
```
