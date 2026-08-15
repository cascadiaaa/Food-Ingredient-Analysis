import re
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model             import Ridge
from sklearn.metrics.pairwise         import cosine_similarity
from sklearn.decomposition            import NMF
from sklearn.cluster                  import KMeans
from sklearn.model_selection          import train_test_split, cross_val_score
from sklearn.metrics                  import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")



# UTILITY

def _float(val) -> float:
    """Extract a numeric value from strings like '72g', '9.00 mg'."""
    try:
        return float(re.sub(r"[^0-9.]", "", str(val)))
    except (ValueError, TypeError):
        return 0.0


# STEP 1 — LOAD DATASETS

print("=" * 70)
print("  FOOD HEALTH SCORER — NLP PIPELINE + RIDGE REGRESSION")
print("=" * 70)

print("\n[1/7] Loading datasets …")

FSSAI_PATH = "FSSAI_Ingredients_Simplified_INS.xlsx"
NUTR_PATH  = "nutrition.xlsx"

fssai     = pd.read_excel(FSSAI_PATH)
fssai.columns = [c.strip() for c in fssai.columns]

nutrition = pd.read_excel(NUTR_PATH)
nutrition.columns = [c.strip().lower().replace(" ", "_") for c in nutrition.columns]

print(f"   → FSSAI rows       : {len(fssai)}")
print(f"   → Nutrition rows   : {len(nutrition)}")


# STEP 2 — FSSAI ADDITIVE ONTOLOGY  (model3.py)

print("\n[2/7] Building FSSAI additive ontology …")

CATEGORY_IMPACT = {
    "Nutrient": 0.247, "Nutrients & Fortification": 0.247,
    "Natural colours": 0.038, "Antioxidant": 0.070,
    "Antioxidant Synergist": 0.028, "Leavening agents": 0.00238,
    "Acidity regulators": 0.00391,
    "Acidifying Agents singly or in combination": 0.00391,
    "Dough conditioners": 0.01172, "Improver": 0.01172,
    "Antifoaming agents": 0.00142, "Jellifying agents": 0.00536,
    "Thickening Agents": 0.00835, "Stabilizers": 0.00835,
    "Emulsifying agents": 0.01963,
    "Emulsifying and stabilizing agents singly or in\ncombination": 0.01963,
    "Structure & Texture Modifiers": 0.01963,
    "Flavours": 0.00995, "Flavor & Aroma": 0.00995,
    "Artificial sweeteners (Singly)": 0.01882, "Sweeteners": 0.00995,
    "COLOURS (Can be used singly or in combination within the specified limits)": 0.00529,
    "Colours (can be used singly or in combination within\nthe specified limits)": 0.00529,
    "Preservatives/ Mould inhibitors singly or in\ncombination": 0.0669,
    "Preservatives (Singly or in combination)": 0.0669,
    "Preservatives & Anti-Microbial Agents": 0.0669,
}

def get_impact(row) -> float:
    for col in ["Sub-Category", "FSSAI Functional Category", "Semantic Group"]:
        val = str(row.get(col, "")).strip()
        if val in CATEGORY_IMPACT:
            return CATEGORY_IMPACT[val]
    return 0.50

fssai["impact_score"] = fssai.apply(get_impact, axis=1)
fssai["harm_score"]   = 1 - fssai["impact_score"]

ins_map   = {}   # INS number  → harm score
name_harm = {}   # name phrase → harm score

for _, row in fssai.iterrows():
    for token in re.findall(r"\d{3,4}(?:\([ivx]+\))?", str(row["INS Number"])):
        ins_map[token] = row["harm_score"]
    for col in ["Ingredient Name", "Simplified Name"]:
        name = re.sub(r"[^a-z0-9 ]", " ", str(row[col]).lower()).strip()
        tokens = name.split()
        if tokens:
            key = " ".join(tokens[:3])
            name_harm[key] = max(name_harm.get(key, 0), row["harm_score"])

print(f"   → {len(ins_map)} INS codes, {len(name_harm)} name phrases indexed")


# STEP 3 — SYNONYM MAP + TEXT NORMALISER  (shared)

SYNONYMS = {
    r"\bmaida\b":                      "refined_flour",
    r"\brefined wheat flour\b":        "refined_flour",
    r"\bwhole wheat flour\b":          "whole_grain",
    r"\bwhole grain\b":                "whole_grain",
    r"\batta\b":                       "whole_grain",
    r"\bcorn starch\b":                "starch",
    r"\bmodified starch\b":            "modified_starch",
    r"\bsucrose\b":                    "added_sugar",
    r"\bglucose syrup\b":              "added_sugar",
    r"\bcorn syrup\b":                 "added_sugar",
    r"\bfructose\b":                   "added_sugar",
    r"\bdextrose\b":                   "added_sugar",
    r"\binvert sugar\b":               "added_sugar",
    r"\bmalt extract\b":               "added_sugar",
    r"\bpalm oil\b":                   "palm_oil",
    r"\bhydrogenated vegetable oil\b": "trans_fat",
    r"\bpartially hydrogenated\b":     "trans_fat",
    r"\bvegetable oil\b":              "vegetable_oil",
    r"\bsunflower oil\b":              "vegetable_oil",
    r"\bolive oil\b":                  "healthy_fat",
    r"\bsodium benzoate\b":            "preservative",
    r"\bpotassium sorbate\b":          "preservative",
    r"\bins 211\b":                    "preservative",
    r"\bins 202\b":                    "preservative",
    r"\bins 282\b":                    "mold_inhibitor",
    r"\btartrazine\b":                 "artificial_colour",
    r"\bins 102\b":                    "artificial_colour",
    r"\bins 110\b":                    "artificial_colour",
    r"\bcaramel colou?r\b":            "caramel_colour",
    r"\bmonosodium glutamate\b":       "msg",
    r"\bmsg\b":                        "msg",
    r"\bins 621\b":                    "msg",
    r"\bins 627\b":                    "flavour_enhancer",
    r"\bins 631\b":                    "flavour_enhancer",
    r"\bsoy lecithin\b":               "emulsifier",
    r"\blecithin\b":                   "emulsifier",
    r"\bins 322\b":                    "emulsifier",
    r"\bins 471\b":                    "emulsifier",
    r"\bskimmed milk\b":               "milk_protein",
    r"\bwhey protein\b":               "protein",
    r"\bsoy protein\b":                "protein",
    r"\boat bran\b":                   "dietary_fibre whole_grain",
    r"\bpsyllium\b":                   "dietary_fibre",
    r"\binulin\b":                     "dietary_fibre prebiotic",
    r"\bflaxseed\b":                   "healthy_fat dietary_fibre",
    r"\bchia\b":                       "healthy_fat dietary_fibre",
}

def normalise(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"\d+\.?\d*\s*%", " ", t)
    t = re.sub(r"\(\s*e\s*\d+\s*\)", " ", t)
    for pattern, replacement in SYNONYMS.items():
        t = re.sub(pattern, " " + replacement + " ", t)
    t = re.sub(r"[^a-z0-9_ ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# STEP 4 — BUILD CORPUS & TF-IDF  (shared vectoriser for both models)


print("\n[3/7] Building corpus and TF-IDF vectoriser …")

# ----- corpus helpers -----

MICRONUTRIENT_THRESHOLDS = {
    "vitamin_c": 20, "vitamin_d": 2, "calcium": 100,
    "irom": 3, "potassium": 300, "zink": 2, "vitamin_b12": 1,
}
MICRONUTRIENT_TOKENS = {
    "vitamin_c":   "vitamins vitamin_c antioxidant immunity",
    "vitamin_d":   "vitamins vitamin_d bone_health",
    "calcium":     "calcium minerals bone_health milk_protein",
    "irom":        "iron minerals",
    "potassium":   "potassium electrolyte minerals heart_health",
    "zink":        "zinc minerals immunity",
    "vitamin_b12": "vitamins vitamin_b12",
}

def nutrition_row_to_tokens(row) -> str:
    """Convert a nutrition row to semantic ingredient-profile tokens."""
    tokens = []
    name_clean = re.sub(r"[^a-z0-9 ]", " ", str(row.get("name", "")).lower())
    tokens.append(name_clean)

    protein  = _float(row.get("protein", 0))
    fat      = _float(row.get("fat", row.get("total_fat", 0)))
    sat_fat  = _float(row.get("saturated_fat", 0))
    sugars   = _float(row.get("sugars", 0))
    fiber    = _float(row.get("fiber", 0))
    sodium   = _float(row.get("sodium", 0))
    calories = _float(row.get("calories", 0))
    chol     = _float(row.get("cholesterol", 0))
    trans    = _float(row.get("fatty_acids_total_trans", 0))

    if protein >= 15:  tokens.append("high_protein protein")
    elif protein >= 5: tokens.append("protein")
    if fat >= 20:      tokens.append("high_fat energy_dense")
    elif fat < 3:      tokens.append("low_fat")
    if sat_fat >= 10:  tokens.append("high_saturated_fat saturated_fat")
    if sugars >= 20:   tokens.append("added_sugar high_sugar")
    elif sugars < 2:   tokens.append("low_sugar")
    if fiber >= 5:     tokens.append("dietary_fibre high_fibre whole_grain gut_health")
    elif fiber >= 2:   tokens.append("dietary_fibre")
    if sodium >= 600:  tokens.append("high_sodium preservative processed_food")
    elif sodium < 50:  tokens.append("low_sodium")
    if calories >= 400: tokens.append("high_calorie energy_dense")
    elif calories < 40: tokens.append("low_calorie")
    if chol >= 100:    tokens.append("high_cholesterol")
    if trans >= 0.5:   tokens.append("trans_fat ultra_processed")

    for nutrient, threshold in MICRONUTRIENT_THRESHOLDS.items():
        if _float(row.get(nutrient, 0)) >= threshold:
            tokens.append(MICRONUTRIENT_TOKENS[nutrient])

    return " ".join(tokens)

# Build combined corpus
fssai_corpus     = [
    normalise(str(r["Ingredient Name"]) + " " + str(r["Simplified Name"]))
    for _, r in fssai.iterrows()
]
nutrition_corpus = [nutrition_row_to_tokens(r) for _, r in nutrition.iterrows()]
training_corpus  = [t for t in fssai_corpus + nutrition_corpus if len(t.strip()) > 5]

print(f"   → FSSAI docs     : {len(fssai_corpus)}")
print(f"   → Nutrition docs : {len(nutrition_corpus)}")
print(f"   → Total corpus   : {len(training_corpus)}")

# Single shared TF-IDF vectoriser
tfidf = TfidfVectorizer(
    ngram_range  = (1, 2),
    min_df       = 2,
    max_df       = 0.95,
    sublinear_tf = True,
    token_pattern= r"[a-z][a-z0-9_]+",
)
tfidf.fit(training_corpus)
vocab_list = tfidf.get_feature_names_out()
print(f"   → Vocabulary size : {len(vocab_list)} terms")


# STEP 5 — TRAIN RIDGE REGRESSION  (ML.py)

print("\n[4/7] Training Ridge Regression on nutrition corpus …")

def compute_health_score(row) -> float:
    """Composite nutrient health score (mirrors ML.py logic)."""
    protein   = _float(row.get("protein", 0))
    fat       = _float(row.get("fat", row.get("total_fat", 0)))
    sat_fat   = _float(row.get("saturated_fat", 0))
    sugars    = _float(row.get("sugars", 0))
    fiber     = _float(row.get("fiber", 0))
    sodium    = _float(row.get("sodium", 0))
    chol      = _float(row.get("cholesterol", 0))
    trans     = _float(row.get("fatty_acids_total_trans", 0))
    vit_c     = _float(row.get("vitamin_c", 0))
    calcium   = _float(row.get("calcium", 0))
    iron      = _float(row.get("irom", 0))
    potassium = _float(row.get("potassium", 0))

    return (
        + protein   * 0.30
        + fiber     * 1.00
        + vit_c     * 0.05
        + calcium   * 0.01
        + iron      * 0.10
        + potassium * 0.003
        - sugars    * 0.30
        - sat_fat   * 0.50
        - trans     * 2.00
        - sodium    * 0.005
        - chol      * 0.02
        - fat       * 0.05
    )

# Build Ridge features & targets from nutrition data
ridge_docs    = [nutrition_row_to_tokens(r) for _, r in nutrition.iterrows()]
ridge_targets = np.array([compute_health_score(r) for _, r in nutrition.iterrows()])

# Normalise targets to [1, 10]
t_min, t_max      = ridge_targets.min(), ridge_targets.max()
ridge_targets_norm = 1 + 9 * (ridge_targets - t_min) / (t_max - t_min + 1e-9)

X_ridge = tfidf.transform(ridge_docs)
y_ridge = ridge_targets_norm

X_train, X_test, y_train, y_test = train_test_split(
    X_ridge, y_ridge, test_size=0.20, random_state=42
)

ridge_model = Ridge(alpha=100.0)
ridge_model.fit(X_train, y_train)

# Performance metrics
y_pred = ridge_model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv   = cross_val_score(ridge_model, X_ridge, y_ridge, cv=5, scoring="r2", n_jobs=-1)

print(f"   → Train samples   : {X_train.shape[0]}")
print(f"   → Test  samples   : {X_test.shape[0]}")
print(f"   → R² Score        : {r2:.4f}")
print(f"   → MAE             : {mae:.4f}")
print(f"   → RMSE            : {rmse:.4f}")
print(f"   → CV R² (5-fold)  : {cv.mean():.4f} ± {cv.std():.4f}")

# Persist normalisation bounds for inference
ridge_t_min, ridge_t_max = t_min, t_max


# STEP 6 — NMF TOPICS + KMEANS CLUSTERS  (model3.py)

print("\n[5/7] NMF topics + KMeans clusters …")

N_TOPICS = 6
fssai_vecs_nmf = tfidf.transform(fssai_corpus)
nmf_model      = NMF(n_components=N_TOPICS, random_state=42, max_iter=500)
nmf_model.fit(fssai_vecs_nmf)
H = nmf_model.components_

TOPIC_NAMES = []
print("\n   Discovered ingredient themes:")
for t in range(N_TOPICS):
    top_idx = H[t].argsort()[::-1][:8]
    words   = [vocab_list[i] for i in top_idx]
    label   = f"Topic {t+1}: {', '.join(words[:5])}"
    TOPIC_NAMES.append(label)
    print(f"   {label}")

N_CLUSTERS  = 6
SAMPLE_SIZE = min(2000, len(training_corpus))
rng         = np.random.default_rng(42)
sample_idx  = rng.choice(len(training_corpus), SAMPLE_SIZE, replace=False)
sample_docs = [training_corpus[i] for i in sample_idx]
sample_vecs = tfidf.transform(sample_docs)

km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
km.fit(sample_vecs)
print(f"\n   KMeans fitted ({N_CLUSTERS} clusters on {SAMPLE_SIZE}-doc sample).")


# STEP 7 — NLP ARCHETYPE VECTORS  (model3.py)

HEALTHY_ARCHETYPE = normalise(
    "whole grain oats dietary fibre protein healthy fat nuts seeds "
    "vitamins minerals calcium iron folate vitamin c whole wheat atta "
    "milk protein live cultures probiotic vegetable fruit legume "
    "olive oil flaxseed chia almonds low sodium no added sugar"
)
UNHEALTHY_ARCHETYPE = normalise(
    "refined flour maida added sugar glucose syrup corn syrup palm oil "
    "trans fat hydrogenated msg flavour enhancer artificial colour "
    "artificial flavour preservative sodium benzoate tartrazine "
    "caramel colour modified starch emulsifier high sodium high sugar "
    "mold inhibitor bleaching agent"
)
healthy_vec   = tfidf.transform([HEALTHY_ARCHETYPE])
unhealthy_vec = tfidf.transform([UNHEALTHY_ARCHETYPE])

print("\n[6/7] Pipeline ready.")
print("=" * 70)


# SCORING FUNCTIONS

def analyse_additives(text: str):
    """Return (weighted_harm_score, count_of_detected_additives)."""
    if not text:
        return 0.0, 0
    t = str(text).lower()
    detected = []
    for token in re.findall(r"ins\s*(\d{3,4}(?:\([ivx]+\))?)", t):
        if token in ins_map:
            detected.append(ins_map[token])
    for phrase, harm in name_harm.items():
        if phrase in t and harm > 0.3:
            detected.append(harm)
    if not detected:
        return 0.0, 0
    return round(np.mean(detected) * 0.6 + max(detected) * 0.4, 3), len(detected)


def hlabel(s: float) -> str:
    if s >= 8:   return "Excellent"
    if s >= 6.5: return "Good"
    if s >= 5:   return "Moderate"
    if s >= 3:   return "Poor"
    return "Very Poor"


def score_product(product_name: str, ingredient_text: str) -> dict:
    """
    Score any food product from its ingredient list.

    Combines:
      • NLP Pipeline score  (cosine similarity + FSSAI penalty)
      • Ridge Regression score  (trained on nutrition TF-IDF features)

    Final score = 0.50 × nlp_score + 0.50 × ridge_score
    Clamped to [1.0, 10.0].

    Parameters
    ----------
    product_name    : str — display name
    ingredient_text : str — raw ingredient list as printed on the pack

    Returns
    -------
    dict with full breakdown
    """
    norm_text   = normalise(ingredient_text)
    product_vec = tfidf.transform([norm_text])

    # ── NLP pipeline ──────────────────────────────────────────────────────
    s_healthy   = float(cosine_similarity(product_vec, healthy_vec)[0, 0])
    s_unhealthy = float(cosine_similarity(product_vec, unhealthy_vec)[0, 0])
    raw_nlp     = s_healthy - s_unhealthy
    nlp_score   = round((raw_nlp + 1) / 2 * 9 + 1, 2)   # → [1, 10]

    penalty, additive_count = analyse_additives(ingredient_text)
    nlp_score_penalised = round(max(1.0, min(10.0, nlp_score * (1 - penalty * 0.4))), 2)

    # ── Ridge Regression ──────────────────────────────────────────────────
    ridge_raw   = float(ridge_model.predict(product_vec)[0])
    ridge_score = round(max(1.0, min(10.0, ridge_raw)), 2)

    # ── Ensemble final score ───────────────────────────────────────────────
    final_score = round(max(1.0, min(10.0,
        0.50 * nlp_score_penalised + 0.50 * ridge_score
    )), 2)

    # ── NMF + KMeans ──────────────────────────────────────────────────────
    topic_weights = nmf_model.transform(product_vec)[0]
    dominant_t    = int(topic_weights.argmax())
    cluster       = int(km.predict(product_vec.toarray())[0]) + 1

    # ── Top recognised tokens ─────────────────────────────────────────────
    nonzero_pairs = [
        (vocab_list[i], product_vec[0, i]) for i in product_vec.nonzero()[1]
    ]
    top_tokens = ", ".join(
        w for w, _ in sorted(nonzero_pairs, key=lambda x: -x[1])[:8]
    ) or "(no recognised tokens)"

    return {
        "product":             product_name,
        # Component scores
        "nlp_score":           nlp_score,
        "nlp_score_penalised": nlp_score_penalised,
        "ridge_score":         ridge_score,
        # NLP details
        "sim_healthy":         round(s_healthy,   4),
        "sim_unhealthy":       round(s_unhealthy, 4),
        "additive_penalty":    penalty,
        "additives_found":     additive_count,
        # Clustering / topics
        "nmf_topic":           TOPIC_NAMES[dominant_t],
        "cluster":             cluster,
        # Final output
        "final_score":         final_score,
        "health_label":        hlabel(final_score),
        "top_tokens":          top_tokens,
    }


def print_result(r: dict):
    score_reduction_pct = round(r["additive_penalty"] * 0.4 * 100, 1)
    print(f"\n{'─'*65}")
    print(f"  Product          : {r['product']}")
    print(f"{'─'*65}")
    print(f"  NLP Raw Score    : {r['nlp_score']} / 10")
    if r["additives_found"] > 0:
        print(f"  Additives Found  : {r['additives_found']}  "
              f"(penalty {r['additive_penalty']}, reduced by {score_reduction_pct}%)")
        print(f"  NLP After Penalty: {r['nlp_score_penalised']} / 10")
    else:
        print(f"  Additives Found  : none detected")
        print(f"  NLP Score        : {r['nlp_score_penalised']} / 10")
    print(f"  Ridge Score      : {r['ridge_score']} / 10")
    print(f"{'─'*65}")
    print(f"  ★ FINAL SCORE    : {r['final_score']} / 10   [{r['health_label']}]")
    print(f"{'─'*65}")
    print(f"  Sim → Healthy    : {r['sim_healthy']}")
    print(f"  Sim → Unhealthy  : {r['sim_unhealthy']}")
    print(f"  Cluster          : {r['cluster']}")
    print(f"  Dominant Topic   : {r['nmf_topic']}")
    print(f"  Key Tokens Seen  : {r['top_tokens']}")


# INTERACTIVE LOOP

print("\n[7/7] Ready.")
print("\n" + "=" * 65)
print("  INTERACTIVE SCORING  (NLP + Ridge Regression ensemble)")
print("  Type 'quit' to exit.")
print("=" * 65)

while True:
    ingredient_text = input("\nEnter ingredients list   : ").strip()
    if ingredient_text.lower() in ("quit", "exit", "q"):
        print("Exiting. Goodbye!")
        break
    if not ingredient_text:
        print("  (no ingredients entered, skipping)")
        continue
    # Use the first 40 chars of ingredients as the display name
    product_name = ingredient_text[:40].rstrip(",; ") + ("…" if len(ingredient_text) > 40 else "")
    result = score_product(product_name, ingredient_text)
    print_result(result)
