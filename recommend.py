import json
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import math, os

# ==========================================
# 1. 평가용 글로벌 음식 베이스라인 (취향 분석용)
# ==========================================
baseline_foods = {
    "맥앤치즈 (Mac & Cheese)": {"taste": [1, 4, 5, 1], "texture": [1, 2, 4, 4, 1]},
    "하리보 젤리 (Gummy Bear)": {"taste": [4, 1, 1, 3], "texture": [1, 5, 2, 1, 1]},
    "나초칩 & 살사 (Nachos)": {"taste": [1, 4, 1, 3], "texture": [5, 1, 1, 2, 1]},
    "페페로니 피자 (Pizza)": {"taste": [2, 5, 4, 2], "texture": [2, 3, 3, 4, 1]}
}
rating_weights = {"1": -1.5, "2": 0.0, "3": 1.0, "4": 2.0}

# ==========================================
# 2. 유저 맞춤형 '찐친' 멘트 사전
# ==========================================
situation_comments = {
    "스트레스": "스트레스 팍팍 받는 날엔 이거 먹고 땀 한 번 쭉 빼면 극락이지!",
    "야식": "이 시간에 먹어도 후회 없을 완벽한 꿀조합! 내일 붓는 건 내일 생각하자.",
    "해장": "어제 달린 네 쓰린 속을 싹 씻어내려 줄 구세주 같은 조합 대령이요~",
    "다이어트": "다이어트 중에 입 터졌을 때? 이 조합이면 죄책감 덜고 맛있게 먹을 수 있어!",
    "안주": "혼술할 때 이거 하나면 이자카야 부럽지 않은 극강의 안주 완성!",
    "가성비": "지갑 얇은 날에도 배 터지고 맛있게 먹을 수 있는 갓성비 조합이야.",
    "디저트": "단 거 당길 때 이거 먹으면 입 안에서 폭죽 터진다? 무조건 담아!"
}

# ==========================================
# 3. 저작권 우회용 말투 변환기 (Paraphraser)
# ==========================================
def paraphrase_snippet(text):
    if not text:
        return "이건 진짜 말로 다 표현 못 할 극락의 맛이야!"
    replacements = {
        "습니다": "어!", "입니다": "이야!", "어요": "어~", "아요": "아~",
        "제가": "내가", "저희는": "우리는", "추천합니다": "무조건 먹어봐!",
        "추천드려요": "완전 강추해!", "요.": "어.", "죠.": "지.",
        "된답니다": "돼!", "바랍니다": "해봐!", "세요": "봐"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    sentences = text.split('.')
    if len(sentences) > 2:
        text = sentences[0] + ". " + sentences[1].strip() + "... (이하 생략)"
    return f"이거 진짜 꿀팁인데, {text.strip()}"

# ==========================================
# 4. 데이터 전처리 파이프라인
# ==========================================
def preprocess_and_load_db(db_filepath='s2_labeled_database.json'):
    if not os.path.exists(db_filepath):
        return [], []
    with open(db_filepath, 'r', encoding='utf-8') as f:
        raw_db = json.load(f)
    clean_db = [item for item in raw_db if 'taste' in item and 'texture' in item]
    top_situations = list(set([sit for item in clean_db for sit in item.get("situations", [])]))
    return clean_db, top_situations

def filter_invalid_data(food_db):
    # 🚫 제외할 키워드 목록
    blacklist_keywords = [
        "오사카", "일본", "로손", "세븐일레븐", "사케", "고량주",
        "위스키", "보드카", "와인", "부산", "중앙해장", "가산디지털단지",
        "김량장동", "안양동", "원흥역"
    ]

    clean_db = []
    for item in food_db:
        name = item.get("combination_name", "")
        snippet = item.get("original_snippet", "")

        # 이름이나 설명에 블랙리스트 키워드가 포함되면 제외
        if any(bad_word in name or bad_word in snippet for bad_word in blacklist_keywords):
            continue

        clean_db.append(item)

    return clean_db


# ==========================================
# 5. 사용자 벡터 생성
# ==========================================
def build_user_vector(spicy_level, ratings):
    spicy_limit = 2 if spicy_level == '1' else (3 if spicy_level == '2' else 5)
    user_vectors = []
    for food_name, features in baseline_foods.items():
        rating = ratings.get(food_name, "2")
        weight = rating_weights[rating]
        food_vector = np.array(features["taste"] + features["texture"])
        user_vectors.append(food_vector * weight)
    user_vector = np.sum(user_vectors, axis=0)
    v_min, v_max = np.min(user_vector), np.max(user_vector)
    if v_max - v_min == 0:
        normalized_user_vector = np.zeros_like(user_vector, dtype=float)
    else:
        normalized_user_vector = (user_vector - v_min) / (v_max - v_min)
    return spicy_limit, normalized_user_vector

# ==========================================
# 6. 추천 알고리즘
# ==========================================
def recommend_kfood(food_db, spicy_limit, situation, user_vector):
    candidates = []
    for food in food_db:
        if food.get("spiciness", 1) > spicy_limit:
            continue

        taste, texture = food["taste"], food["texture"]
        food_vector = np.array([
            taste.get("sweet", 1), taste.get("salty", 1), taste.get("rich", 1), taste.get("sour", 1),
            texture.get("crispy", 1), texture.get("chewy", 1), texture.get("soft", 1),
            texture.get("thick", 1), texture.get("popping", 1)
        ])

        # ✅ sim 정의
        if np.linalg.norm(user_vector) > 0 and np.linalg.norm(food_vector) > 0:
            sim = cosine_similarity(user_vector.reshape(1, -1), food_vector.reshape(1, -1))[0][0]
        else:
            sim = 0

        if situation and situation in food.get("situations", []):
            sim += 0.5

        raw_snippet = food.get("original_snippet", "")
        safe_snippet = paraphrase_snippet(raw_snippet)

        # ✅ sim을 이용해 match_rate 계산
        match_rate = math.ceil(min(max(sim * 50 + 50, 0), 100))

        candidates.append({
            "name": food.get("combination_name", "비밀의 꿀조합"),
            "score": match_rate,
            "snippet": safe_snippet
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:3]
