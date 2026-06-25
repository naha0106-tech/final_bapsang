from flask import Flask, render_template, request, redirect, url_for, session
import cv2, os, random, json, numpy as np
import recommend 
from ultralytics import YOLO
from recommend import filter_invalid_data

app = Flask(__name__)
app.secret_key = "secret_key"  # 세션 저장용

# 1. 모델, DB, 매핑 파일 로드
model = YOLO('best.pt')
with open("samgyeopsal_recipe_db_merged!.json", "r", encoding="utf-8") as f:
    s1_database = json.load(f)
with open("food_mapping.json", "r", encoding="utf-8") as f:
    food_map = json.load(f)

ALLERGY_MAP = {"견과류": 5, "땅콩": 6, "갑각류": 7, "생선": 8, "밀": 9, "달걀": 10, "유제품": 11, "대두": 12, "참깨": 13}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start")
def start():
    return render_template("start.html")



# 한술 버튼 → 알러지 설문 페이지
@app.route("/survey", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        allergies = request.form.getlist("allergy")
        safety_mask = np.zeros(14, dtype=int)
        for allergy in allergies:
            if allergy in ALLERGY_MAP:
                safety_mask[ALLERGY_MAP[allergy]] = 1
        session["safety_mask"] = safety_mask.tolist()
        return redirect(url_for("onespoon"))
    return render_template("survey.html", allergies=list(ALLERGY_MAP.keys()))

# 알러지 반영 후 카메라 촬영 + YOLO 분석
@app.route("/onespoon")
def onespoon():
    safety_mask = np.array(session.get("safety_mask", np.zeros(14, dtype=int)))

    cap = cv2.VideoCapture(2)  # USB 카메라 (환경에 맞게 번호 조정)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "카메라를 열 수 없습니다."

    results = model(frame)
    detected_korean = {model.names[int(box.cls[0])] for r in results for box in r.boxes}

    # 한국어 → 영어 매핑
    detected_english_set = set()
    for k_item in detected_korean:
        mapped = food_map.get(k_item, [k_item])
        if isinstance(mapped, list):
            detected_english_set.update([m.strip().lower() for m in mapped])
        else:
            detected_english_set.add(mapped.strip().lower())

    # 레시피 매칭 (레시피의 모든 재료가 감지된 음식에 포함될 때만)
    found_recipes = []
    for entry in s1_database:
        recipe_ingredients = [m.strip().lower() for m in entry.get('main_ingredients', []) + entry.get('sub_ingredients', [])]
        if recipe_ingredients and all(m in detected_english_set for m in recipe_ingredients):
            # 알러지 체크
            if not any(allergy in recipe_ingredients
                       for allergy, idx in ALLERGY_MAP.items() if safety_mask[idx] == 1):
                found_recipes.append(entry)

    cap.release()
    img_path = os.path.join("static", "capture_result.jpg")
    cv2.imwrite(img_path, results[0].plot())

    # 3개 이상이면 랜덤으로 3개만 선택
    if len(found_recipes) > 3:
        found_recipes = random.sample(found_recipes, 3)

    return render_template("result.html",
                           image_url=url_for('static', filename='capture_result.jpg'),
                           recipes=found_recipes)


@app.route("/twospoon", methods=["GET", "POST"])
def twospoon():
    if request.method == "POST":
        spicy_level = request.form.get("spicy_level", "2")
        situation = request.form.get("situation", "")
        ratings = {food: request.form.get(food, "2") for food in recommend.baseline_foods.keys()}

       # 1️⃣ DB 불러오기
        food_db, top_situations = recommend.preprocess_and_load_db('s2_labeled_database.json')

        # 2️⃣ 필터링 적용
        food_db = filter_invalid_data(food_db)

        # 3️⃣ 추천 실행
        spicy_limit, user_vector = recommend.build_user_vector(spicy_level, ratings)
        
        results = recommend.recommend_kfood(food_db, spicy_limit, situation, user_vector)
        


        return render_template("result_two.html", recipes=results, situation=situation)

    return render_template("survey_two.html", foods=recommend.baseline_foods.keys())

if __name__ == "__main__":
    app.run(debug=True)

