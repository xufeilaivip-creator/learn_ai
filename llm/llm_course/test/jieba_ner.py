import jieba
import jieba.posseg as pseg


def recognize_entities(text):
    """
    使用jieba进行命名实体识别
    识别人名(nr)、地名(ns)、机构名(nt)
    """
    # 加载词典提高识别准确率
    jieba.load_userdict("dict.txt")  # 可以自定义词典提高识别效果

    # 进行词性标注
    words = pseg.cut(text)

    entities = []
    for word, flag in words:
        # 判断词性并分类
        if flag == 'nr':
            entities.append((word, "人名"))
        elif flag == 'ns':
            entities.append((word, "地名"))
        elif flag == 'nt':
            entities.append((word, "机构名"))

    return entities


if __name__ == "__main__":
    text = "北京是中国的首都，马云是阿里巴巴创始人"
    result = recognize_entities(text)

    print("识别结果：")
    for entity, entity_type in result:
        print(f"{entity}（{entity_type}）")
