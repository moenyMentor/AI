from model import load_llm_model
from sys_prompt import guide_prompt

#사용자 입력받을 변수
age = 40
salary = "7000만원"
goal =  "10년 내에 5억 벌고 은퇴하고 싶습니다."
seed_money =  "2000만원"
investment_exp = "주식 투자 경험 있음"
investment_tendency = "공격투자형"
available_amount = "350만원"

prompt = guide_prompt(age, salary, goal, seed_money, investment_exp, investment_tendency, available_amount)
llm = load_llm_model()

def create_guide():
    
    response = llm.generate(prompts=[prompt])

    return response.generations[0][0].text

if __name__ == "__main__":
    create_guide()