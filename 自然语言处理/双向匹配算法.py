import numpy as np


name = '研究生命的起源我热爱大自然'

def match_yonghu(stg):
    if stg[:2] == '用户':
        return True
    else:
        return False
def is_endwithxinxi(stg):
    if stg[len(stg)-2:] == '信息':
        return True
    else:
        return False


x1 = match_yonghu(name)
x2 = is_endwithxinxi(name)
print(x1,x2)

v = np.array([1,2,3],dtype=float)
m = np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])


temp = np.random.randn(2,5)
print(temp.shape,temp)
temp = temp.reshape(1,10)
print(temp.shape,temp)



class Tokenizer_self():
    def __init__(self):
        self.max_size = 3
        self.dictnary = ['研究','命','的','起源','研究生','生命']
        self.ans_forword = []
        self.ans_reverse = []
    def mm(self,text):

        len_text = len(text)
        while len_text > 0:
            divided_text = text[:self.max_size]
            while divided_text not in self.dictnary:
                while len(divided_text) == 1:
                    print(f'{divided_text}----未被匹配')
                    break
                divided_text = divided_text[:len(divided_text)-1]
            self.ans_forword.append(divided_text)
            text = text[len(divided_text):]
            len_text = len(text)
        # print(f'正向匹配分词结果：{self.ans_forword}')
        return self.ans_forword

    def rmm(self,text):

        len_text = len(text)
        while len_text > 0:
            divided_text = text[-self.max_size:]   #获取末尾的词

            while divided_text not in self.dictnary:    #若当前词不在字典里则需要去除首字符继续匹配 若词长为1 仍然不在词典里 则未匹配到
                if len(divided_text) == 1:
                    print(f"{divided_text}----未被匹配")
                    break
                divided_text = divided_text[-(len(divided_text)-1):]   #去除首字符
            self.ans_reverse.append(divided_text)                      #添加匹配成功的字符

            text = text[:-len(divided_text)]
            len_text = len(text)
        # print(f'逆向匹配分词结果：{self.ans_reverse[::-1]}')
        return self.ans_reverse[::-1]
    def double_mm(self,text):
        self.ans_forword = []
        self.ans_reverse = []
        result_for = self.mm(text)
        result_re = self.rmm(text)
        if len(result_for) > len(result_re):
            return result_re
        elif len(result_for) < len(result_re):
            return result_for
        else:
            count_for = 0
            count_re = 0
            for word in result_for:
                if len(word) == 1:
                    count_for += 1
            for word in result_re:
                if len(word) == 1:
                    count_re += 1
            if count_for > count_re:    #返回单字数少的那个
                return  result_re
            else:
                return result_for


tokenizer = Tokenizer_self()
rm = tokenizer.mm(name)
rrm = tokenizer.rmm(name)
rdm = tokenizer.double_mm(name)
print(rm,rrm,rdm)
# print(rdm)



