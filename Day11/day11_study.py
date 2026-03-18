class Runnable:
    def invoke(self, input, config=None):
        raise NotImplementedError("invoke方法未实现")

    def __or__(self, other):
        return RunnableSequence(self, other)


class PromptTemplate(Runnable):
    def __init__(self, template):
        self.template = template

    def invoke(self, input_dict, config=None):
        return self.template.format(**input_dict)


class SimpleModel(Runnable):
    def invoke(self, input_dict, config=None):
        return f"模型处理后的结果：{input_dict}"


class RunnableSequence(Runnable):
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    def invoke(self, input, config=None):
        x = self.first.invoke(input, config)
        x = self.second.invoke(x, config)
        return x