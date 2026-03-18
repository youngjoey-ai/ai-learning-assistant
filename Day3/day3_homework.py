def get_text_length(text):
    length = len(text)
    return length

def filter_ai_skills(skills):
    ai_skills = []
    for skill in skills:
        if 'AI' in skill or '大模型' in skill or 'LangChain' in skill:
            ai_skills.append(skill)
    return ai_skills

text = "我要30天学会AI大模型应用开发"
print('文本长度: ', get_text_length(text))

all_skills = ["Python", "AI开发", "Java", "LangChain", "大模型RAG", "前端"]
ai_skills = filter_ai_skills(all_skills)
print('AI相关技能:', ai_skills)