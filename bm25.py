from rank_bm25 import *
import string

def remove_puncts(input_string, string):
    return input_string.translate(str.maketrans('', '', string.punctuation)).lower()

def run_bm25(lst, key):
    # Chuyen tat ca chu cai ve dang chu thuong, xoa cac ki hieu khong can thiet
    lst_new = [remove_puncts(line, string) for line in lst]
    key_new = remove_puncts(key, string)
    # Tao token cho lst va key
    token_lst = [line.split(" ") for line in lst_new]
    token_key = key_new.split(" ")
    print(token_lst)
    print(token_key)
    # Tao doi tuong BM25OKAPI
    bm25 = BM25Okapi(token_lst)
    # Tinh score cho moi line trong lst
    score = bm25.get_scores(token_key)
    print(score)
    # Tao list gom cac cap [score, index]
    list_tmp = [[sc, index] for index, sc in enumerate(score) if sc != 0]
    print(list_tmp)
    list_tmp.sort(reverse=True)
    print(list_tmp)
    
    lst_priority = []
    for obj in list_tmp:
        lst_priority.append(obj[1])
    return lst_priority

if __name__ == "__main__":
    pass


