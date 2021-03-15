from smoothnlp.algorithm.phrase import extract_phrase



sentences = ['为什么我无法看到额度', '为什么开通了却没有额度', '为两次还都提示失败呢','为什么我申请额度输入密码就一直是那个页面', '为什么要输入支付密码来验证','今天借明天还款可以？','借了钱，但还没有通过，可以取消吗？']




def main():
    new_phrases = extract_phrase(sentences)
    print(new_phrases)

if __name__ == '__main__':
    main()



