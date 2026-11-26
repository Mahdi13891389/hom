import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# ۱. تولید دیتاست استاندارد Record Matching & Ranking (train.csv و test.csv)
# ==============================================================================
def generate_lost_found_dataset():
    os.makedirs('data', exist_ok=True)
    if os.path.exists('data/train.csv') and os.path.exists('data/test.csv'):
        return pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')

    categories = {
        'Mobile': ['گوشی آیفون ۱۳ مشکی قاب سیلیکونی', 'موبایل سامسونگ گلکسی آبی صفحه ترک خورده', 'گوشی شیائومی نوت ۱۲ خاکستری'],
        'Wallet': ['کیف پول چرم قهوه ای حاوی کارت بانکی و ملی', 'کیف مدارک مشکی کوچک زیپ دار', 'کیف پول زنانه قرمز چرمی'],
        'Bag': ['کوله پشتی لپ تاپ مشکی لنوو', 'ساک ورزشی آبی نایکی', 'کیف دوشی چرم عسلی'],
        'Keys': ['دسته کلید با جاکلیدی فلزی ماشین', 'تک کلید سوئیچ پژو ریموت دار', 'دسته کلید خانه با آویز قرمز'],
        'Watch': ['ساعت هوشمند اپل واچ بند مشکی', 'ساعت مچی عقربه ای نقره ای کاسیو', 'مچ بند هوشمند شیائومی']
    }
    colors = ['مشکی', 'آبی', 'قهوه ای', 'قرمز', 'خاکستری', 'نقره ای']
    locations = [
        ('ایستگاه مترو صادقیه', 35.7219, 51.3347),
        ('میدان ونک', 35.7575, 51.4100),
        ('ترمینال غرب', 35.7117, 51.3219),
        ('میدان انقلاب', 35.7011, 51.3912),
        ('ایستگاه مترو تجریش', 35.8062, 51.4285),
        ('فرودگاه مهرآباد', 35.6892, 51.3134)
    ]

    def build_pairs(n_queries, candidates_per_query, is_train=True, prefix='TR'):
        rows = []
        base_time = pd.Timestamp('2026-01-10 08:00:00')

        for q in range(n_queries):
            lost_id = f'{prefix}_LOST_{q:04d}'
            cat = np.random.choice(list(categories.keys()))
            desc_true = np.random.choice(categories[cat])
            color_true = next((c for c in colors if c in desc_true), np.random.choice(colors))
            loc_name, lat_l, lon_l = locations[np.random.randint(len(locations))]
            lost_time = base_time + pd.Timedelta(hours=np.random.randint(0, 720))

            # تعیین اینکه کدام کاندیدا تطابق واقعی دارد
            match_idx = np.random.randint(0, candidates_per_query)

            for c in range(candidates_per_query):
                found_id = f'{prefix}_FOUND_{q:04d}_{c:02d}'
                pair_id = f'{lost_id}__{found_id}'

                if c == match_idx:
                    # کاندیدای مطابق واقعی (با کمی تفاوت طبیعی در نگارش و مکان نزدیک)
                    f_cat = cat
                    words = desc_true.split()
                    np.random.shuffle(words)
                    f_desc = ' '.join(words[:max(2, len(words) - 1)]) + ' پیدا شده'
                    f_color = color_true
                    f_lat = lat_l + np.random.normal(0, 0.004)
                    f_lon = lon_l + np.random.normal(0, 0.004)
                    f_loc = loc_name
                    f_time = lost_time + pd.Timedelta(hours=np.random.randint(1, 24))
                    is_match = 1
                else:
                    # کاندیدای نامرتبط یا نیمه‌مشابه (Hard Negative)
                    is_hard_neg = np.random.rand() < 0.35
                    f_cat = cat if is_hard_neg else np.random.choice(list(categories.keys()))
                    f_desc = np.random.choice(categories[f_cat])
                    f_color = np.random.choice(colors)
                    r_loc, r_lat, r_lon = locations[np.random.randint(len(locations))]
                    f_lat = r_lat + np.random.normal(0, 0.01)
                    f_lon = r_lon + np.random.normal(0, 0.01)
                    f_loc = r_loc
                    f_time = lost_time + pd.Timedelta(hours=np.random.randint(-48, 96))
                    is_match = 0

                row = {
                    'pair_id': pair_id,
                    'lost_id': lost_id,
                    'found_id': found_id,
                    'lost_category': cat,
                    'found_category': f_cat,
                    'lost_color': color_true,
                    'found_color': f_color,
                    'lost_desc': desc_true,
                    'found_desc': f_desc,
                    'lost_location': loc_name,
                    'found_location': f_loc,
                    'lost_lat': round(lat_l, 5),
                    'lost_lon': round(lon_l, 5),
                    'found_lat': round(f_lat, 5),
                    'found_lon': round(f_lon, 5),
                    'lost_time': lost_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'found_time': f_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                if is_train:
                    row['is_match'] = is_match
                rows.append(row)

        return pd.DataFrame(rows)

    train_df = build_pairs(n_queries=600, candidates_per_query=6, is_train=True, prefix='TR')
    test_df = build_pairs(n_queries=200, candidates_per_query=6, is_train=False, prefix='TE')

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    return train_df, test_df

# ==============================================================================
# ۲. مهندسی ویژگی‌های تطبیق اطلاعات (Text, Spatial, Temporal Features)
# ==============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def jaccard_similarity(s1, s2):
    set1, set2 = set(str(s1).split()), set(str(s2).split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def extract_matching_features(df, tfidf=None, fit_tfidf=False):
    df = df.copy()

    # ۱. تطابق ویژگی‌های دسته‌ای
    df['category_match'] = (df['lost_category'] == df['found_category']).astype(int)
    df['color_match'] = (df['lost_color'] == df['found_color']).astype(int)
    df['location_name_match'] = (df['lost_location'] == df['found_location']).astype(int)

    # ۲. فاصله جغرافیایی Haversine
    df['distance_km'] = haversine_km(df['lost_lat'], df['lost_lon'], df['found_lat'], df['found_lon'])

    # ۳. اختلاف زمانی و منطق ترتیبی
    t_lost = pd.to_datetime(df['lost_time'])
    t_found = pd.to_datetime(df['found_time'])
    df['time_diff_hours'] = (t_found - t_lost).dt.total_seconds() / 3600.0
    df['abs_time_diff_hours'] = df['time_diff_hours'].abs()
    df['found_after_lost'] = (df['time_diff_hours'] >= 0).astype(int)

    # ۴. شباهت متنی (Jaccard + TF-IDF Cosine Similarity)
    df['jaccard_sim'] = [jaccard_similarity(a, b) for a, b in zip(df['lost_desc'], df['found_desc'])]

    if fit_tfidf:
        tfidf = TfidfVectorizer(ngram_range=(1, 2))
        all_texts = pd.concat([df['lost_desc'], df['found_desc']]).unique()
        tfidf.fit(all_texts)

    v_lost = tfidf.transform(df['lost_desc'])
    v_found = tfidf.transform(df['found_desc'])
    df['tfidf_cosine_sim'] = np.asarray(v_lost.multiply(v_found).sum(axis=1)).ravel()

    return df, tfidf

# ==============================================================================
# ۳. آموزش مدل رتبه‌بندی (GroupKFold) و تولید خروجی استاندارد مسابقه
# ==============================================================================
if __name__ == '__main__':
    train_raw, test_raw = generate_lost_found_dataset()

    train_feat, tfidf_vec = extract_matching_features(train_raw, fit_tfidf=True)
    test_feat, _ = extract_matching_features(test_raw, tfidf=tfidf_vec, fit_tfidf=False)

    feature_cols = [
        'category_match', 'color_match', 'location_name_match',
        'distance_km', 'time_diff_hours', 'abs_time_diff_hours',
        'found_after_lost', 'jaccard_sim', 'tfidf_cosine_sim'
    ]

    X = train_feat[feature_cols]
    y = train_feat['is_match']
    groups = train_feat['lost_id']
    X_test = test_feat[feature_cols]

    # استفاده از GroupKFold برای جلوگیری از نشت رکوردهای یک شیء گمشده بین آموزش و اعتبارسنجی
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42 + fold
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / gkf.n_splits

    roc_auc = roc_auc_score(y, oof_preds)
    pr_auc = average_precision_score(y, oof_preds)

    # محاسبه دقت رتبه اول (Top-1 Ranking Accuracy)
    train_feat['pred_score'] = oof_preds
    top1_hits = 0
    for _, grp in train_feat.groupby('lost_id'):
        best_row = grp.loc[grp['pred_score'].idxmax()]
        if best_row['is_match'] == 1:
            top1_hits += 1
    top1_acc = top1_hits / train_feat['lost_id'].nunique()

    print("-" * 52)
    print("نتایج ارزیابی سیستم هوشمند تطبیق و رتبه‌بندی اشیاء:")
    print(f"ROC-AUC Score:       {roc_auc:.4f}")
    print(f"PR-AUC Score:        {pr_auc:.4f}")
    print(f"Top-1 Ranking Acc:   {top1_acc * 100:.2f}%")
    print("-" * 52)

    # ساخت فایل خروجی رتبه‌بندی شده در استاندارد مسابقات
    test_feat['match_probability'] = np.round(test_preds, 4)
    test_feat['rank'] = test_feat.groupby('lost_id')['match_probability'].rank(ascending=False, method='first').astype(int)
    submission = test_feat.sort_values(['lost_id', 'rank'])[['pair_id', 'lost_id', 'found_id', 'match_probability', 'rank']]
    submission.to_csv('submission.csv', index=False)
    print("فایل submission.csv با موفقیت ساخته شد!")
