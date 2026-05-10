# Fake News Detection - Crypto Domain
## הוראות התקנה והפעלה

**שם:** [שמך] | **ת.ז.:** [מספר זהות]

---

## דרישות מקדימות
- Python 3.9+
- pip

---

## שלבי התקנה

### 1. התקנת ספריות
```bash
pip install -r requirements.txt
```

### 2. הגדרת Credentials

**Reddit API (חינמי לחלוטין):**
1. כנסי ל: https://www.reddit.com/prefs/apps
2. Create Another App -> בחרי "script"
3. Name: FakeNewsDetector
4. Redirect URI: http://localhost:8080
5. צלמי את `client_id` (מתחת לשם) ו-`client_secret`

**הגדרת קובץ `.env`:**
```bash
cp .env.example .env
# פתחי את .env ומלאי את הפרטים
```

### 3. הפעלת Jupyter Notebook
```bash
jupyter notebook notebooks/01_EDA_and_Preprocessing.ipynb
```

---

## מבנה הפרויקט
```
fake-news-crypto/
├── notebooks/
│   └── 01_EDA_and_Preprocessing.ipynb   <- ה-Notebook הראשי
├── data/
│   ├── raw/          <- נתונים גולמיים (נוצרים אוטומטית)
│   └── processed/    <- נתונים מעובדים + גרפים
├── src/              <- קוד עזר (שלב הבא)
├── .env              <- credentials (לא לשמור ב-Git!)
├── .env.example      <- תבנית לקובץ .env
├── requirements.txt
└── SETUP.md
```

---

## מקורות הנתונים

| מקור | כמות | תיאור |
|------|------|--------|
| LIAR Dataset (HuggingFace) | ~12,800 | הצהרות מתויגות 6 רמות |
| Fake/Real News (GitHub) | ~44,000 | כתבות FAKE/REAL |
| Reddit Scraping | ~1,200 | פוסטים מ-r/CryptoCurrency, r/Bitcoin, r/Buttcoin |
