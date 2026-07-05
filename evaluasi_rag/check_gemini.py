"""
Jalankan script ini DARI KOMPUTERMU sendiri:
    python check_gemini.py

Script ini akan menampilkan:
- Model Gemini apa saja yang tersedia di API key kamu
- Mana yang bisa dipakai untuk generateContent
"""
import os, requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    print("❌ GEMINI_API_KEY tidak ditemukan di .env")
    exit(1)

print(f"🔑 API Key  : {API_KEY[:12]}...{API_KEY[-4:]}")
print("=" * 60)

for api_ver in ["v1", "v1beta"]:
    url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        print(f"\n[{api_ver}] HTTP {r.status_code}")

        if r.status_code != 200:
            print(f"  Error: {r.text[:200]}")
            continue

        models = r.json().get("models", [])
        print(f"  Total model: {len(models)}")
        print(f"\n  {'Model Name':<45} {'generateContent':>16}")
        print(f"  {'-'*45} {'-'*16}")

        for m in sorted(models, key=lambda x: x["name"]):
            methods = m.get("supportedGenerationMethods", [])
            can_gen = "✅ YES" if "generateContent" in methods else "❌ no"
            print(f"  {m['name']:<45} {can_gen:>16}")

    except Exception as e:
        print(f"  Exception: {e}")

print("\n" + "=" * 60)
print("Salin nama model yang ✅ YES ke GEMINI_MODEL_CHAIN di llm.py")
print("Format: hilangkan prefix 'models/' → 'gemini-2.0-flash'")
