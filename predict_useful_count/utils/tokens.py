from typing import List, TypedDict, Union
import os
import CaboCha
from dotenv import load_dotenv


current_path = os.path.dirname(os.path.abspath(__file__))

load_dotenv()


class Token(TypedDict):
    """
    形態素を表す型

    Attributes
    ----------
    base: str
        形態素の原型 or 表層型
    surface: str
        形態素の表層型
    pos: str
        品詞
    pos_detail: str
        品詞の詳細
    """

    base: str
    surface: str
    pos: str
    pos_detail: str


def get_evaluation_expressions() -> List[str]:
    """
    「評価値表現辞書」から評価表現を取得し、配列を返す
    - 評価表現の重複は排除

    Returns
    -------
    tokens: List[str]
        形態素の原型 or 表層型となる文字列
    """

    tokens: List[str] = []
    with open(f"{current_path}/../dic/EVALDIC_ver1.01", "r") as f:
        for line in f:
            token = line.strip()
            tokens.append(token)
    return tokens


def conect_compound_words(tokens: List[Token]) -> List[Token]:
    new_tokens: List[Token] = []

    index = 0
    while index < len(tokens):
        if index == len(tokens) - 1:
            new_tokens.append(tokens[index])
            index += 1
            continue
        if (
            tokens[index + 1]["pos"] == "形容詞"
            and tokens[index + 1]["pos_detail"] == "接尾"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": None,
                    "pos": "形容詞",
                    "pos_detail": "自立",
                }
            )
            index += 2
        elif (
            tokens[index + 1]["pos"] == "名詞" and tokens[index + 1]["pos_detail"] == "接尾"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": None,
                    "pos": "名詞",
                    "pos_detail": "自立",
                }
            )
            index += 2
        elif (
            tokens[index + 1]["pos"] == "動詞" and tokens[index + 1]["pos_detail"] == "接尾"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": None,
                    "pos": "動詞",
                    "pos_detail": "自立",
                }
            )
            index += 2
        elif (
            tokens[index]["pos"] == "接頭詞"
            and tokens[index]["pos_detail"] == "名詞接続"
            and tokens[index + 1]["pos"] == "名詞"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "pos": "名詞",
                    "pos_detail": "一般",
                }
            )
            index += 2
        elif (
            tokens[index]["pos"] == "名詞"
            and tokens[index + 1]["pos_detail"] == "ナイ形容詞語幹"
            and tokens[index + 1]["surface"] == "ない"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "pos": "名詞",
                    "pos_detail": "一般",
                }
            )
            index += 2
        elif tokens[index + 1]["pos"] == "助動詞" and tokens[index + 1]["surface"] == "た":
            new_tokens.append(tokens[index])
            index += 2
        else:
            new_tokens.append(tokens[index])
            index += 1

    return new_tokens


def is_used_token(token: Token):
    evaluation_expressions = get_evaluation_expressions()
    if token["pos"] == ["名詞", "形容詞", "動詞", "副詞"]:
        return True
    if (
        token["surface"] in evaluation_expressions
        or token["base"] in evaluation_expressions
    ):
        return True


def get_token_list(sentence: str) -> Union[List[str], None]:
    if sentence == "":
        return None

    cabocha = CaboCha.Parser(os.getenv("NEOLOGD_PATH"))
    tree = cabocha.parse(sentence)

    chunks: List[List[Token]] = []
    tokens: List[Token] = []
    for token_index in range(tree.size()):
        token = tree.token(token_index)
        if token.chunk is not None:
            chunks.append(conect_compound_words(tokens=tokens)) if len(tokens) else None
            tokens = []
        token_feature = token.feature.split(",")
        surface = token.surface
        base = token_feature[6] if token_feature[6] != "*" else None
        pos = token_feature[0]
        pos_detail = token_feature[1]
        tokens.append(
            {
                "surface": surface,
                "base": base,
                "pos": pos,
                "pos_detail": pos_detail,
            }
        )
    chunks.append(conect_compound_words(tokens=tokens))

    token_list = [token for chunk in chunks for token in chunk]
    filtered_token_list = list(filter(is_used_token, token_list))

    return [
        token["base"] if token["base"] else token["surface"]
        for token in filtered_token_list
    ]
