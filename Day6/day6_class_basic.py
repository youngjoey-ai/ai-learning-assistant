class AILearner:
    def __init__(self, name, study_day):
        self.name = name
        self.study_day = study_day
        self.skills = []
    
    def add_skill(self, skill):
        self.skills.append(skill)
        print(f'{self.name}学会了：{skill}')

    def show_progress(self):
        print(f'\n【{self.name}的学习进度】')
        print(f'学习天数：{self.study_day}天')
        print(f'已掌握技能：{self.skills}')


learner = AILearner("AI程序员", 6)

learner.add_skill("Python基础")
learner.add_skill("文件操作")
learner.add_skill("异常处理")
learner.add_skill("模块调用")

learner.show_progress()