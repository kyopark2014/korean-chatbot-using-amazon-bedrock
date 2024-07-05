# PDF에서 정보 추출하기

## 문서 추출의 어려움

[Why Text Extraction is hard](https://pypdf.readthedocs.io/en/stable/user/extract-text.html#why-text-extraction-is-hard)에서는 텍스트 추출의 어려움에 대해 설명하고 있습니다.

- Paragraphs: Should the text of a paragraph have line breaks at the same places where the original PDF had them or should it rather be one block of text?

- Page numbers: Should they be included in the extract?

- Headers and Footers: Similar to page numbers - should they be extracted?

- Outlines: Should outlines be extracted at all?

- Formatting: If text is bold or italic, should it be included in the output?

- Tables: Should the text extraction skip tables? Should it extract just the text? Should the borders be shown in some Markdown-like way or should the structure be present e.g. as an HTML table? How would you deal with merged cells?

- Captions: Should image and table captions be included?

- Ligatures: The Unicode symbol U+FB00 is a single symbol ﬀ for two lowercase letters ‘f’. Should that be parsed as the Unicode symbol ‘ﬀ’ or as two ASCII symbols ‘ff’?

- SVG images: Should the text parts be extracted?

- Mathematical Formulas: Should they be extracted? Formulas have indices, and nested fractions.

- Whitespace characters: How many new lines should be extracted for 3cm of vertical whitespace? How many spaces should be extracted if there is 3cm of horizontal whitespace? When would you extract tabs and when spaces?

- Footnotes: When the text of multiple pages is extracted, where should footnotes be shown?

- Hyperlinks and Metadata: Should it be extracted at all? Where should it be placed in which format?

- Linearization: Assume you have a floating figure in between a paragraph. Do you first finish the paragraph or do you put the figure text in between?

## Annotationd의 활용

[Some notes about form fields and annotations](https://pypdf.readthedocs.io/en/stable/user/forms.html#some-notes-about-form-fields-and-annotations)와 같은 Annotationd을 가지고 있습니다. 

이미지의 리소스에서 XObject을 이용하면 해당 Page에 이미지가 있음을 알 수 있습니다.

```text
Resources[0]:
{
   "/ProcSet":[
      "/PDF",
      "/Text",
      "/ImageB",
      "/ImageC",
      "/ImageI"
   ],
   "/XObject":{
      "/Im1":IndirectObject(7,
      0,
      139883439847120),
      "/Im2":IndirectObject(9,
      0,
      139883439847120),
      "/Im3":IndirectObject(10,
      0,
      139883439847120)
   }
}
```

생성되는 이미지의 크기는 아래와 같습니다.

```text
MediaBox[1]: [0, 0, 960, 540]
```

