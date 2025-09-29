import spacy


def recognize_entities(text):
    """
    使用spaCy进行命名实体识别
    识别人名、地名、机构名
    """
    # 加载中文模型，首次使用需要先下载：python -m spacy download zh_core_web_sm
    nlp = spacy.load("zh_core_web_sm")

    # 处理文本
    doc = nlp(text)

    entities = []
    # 定义实体类型映射
    entity_map = {
        "PERSON": "人名",
        "GPE": "地名",  # 国家、城市、州等
        "ORG": "机构名"  # 公司、机构、企业等
    }

    for ent in doc.ents:
        if ent.label_ in entity_map:
            entities.append((ent.text, entity_map[ent.label_]))

    return entities


if __name__ == "__main__":
    text = "北京是中国的首都，马云是阿里巴巴创始人"
    result = recognize_entities(text)

    print("识别结果：")
    for entity, entity_type in result:
        print(f"{entity}（{entity_type}）")
