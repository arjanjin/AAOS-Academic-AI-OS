# 📤 วิธี Upload AAOS ขึ้น GitHub

## ขั้นตอนที่ 1 — สร้าง Repository บน GitHub

1. เปิด https://github.com/new
2. กรอกข้อมูล:
   - Repository name: `AAOS-Academic-AI-OS`
   - Description: `Academic AI Operating System — Self-improving multi-agent framework for professors`
   - Public ✅ (เพื่อ citation)
   - Add README: ❌ (เราจะ push เอง)
3. คลิก "Create repository"

---

## ขั้นตอนที่ 2 — เตรียมไฟล์บน Windows

```powershell
# สร้างโฟลเดอร์ repo
mkdir D:\AAOS-Academic-AI-OS
cd D:\AAOS-Academic-AI-OS

# สร้างโครงสร้าง
mkdir agents
mkdir docs
mkdir examples

# Copy ไฟล์ agents
copy D:\arjin-vault\06-OUTPUT\AAOS\agents\engineering_agent.py agents\
copy D:\arjin-vault\06-OUTPUT\AAOS\agents\researcher_agent.py agents\
copy D:\arjin-vault\06-OUTPUT\AAOS\agents\medical_agent.py agents\
copy D:\arjin-vault\06-OUTPUT\AAOS\agents\business_agent.py agents\
copy D:\arjin-vault\06-OUTPUT\AAOS\agents\orchestrator.py agents\
copy D:\arjin-vault\06-OUTPUT\AAOS\agents\langgraph_orchestrator.py agents\

# Copy files จาก download
copy <download_path>\README.md .
copy <download_path>\LICENSE .
copy <download_path>\requirements.txt .
copy <download_path>\.gitignore .
copy <download_path>\.env.template .
```

---

## ขั้นตอนที่ 3 — แก้ไข Path ใน Python Files

ก่อน push ต้องแก้ hardcoded paths ใน agents ให้ใช้ environment variable:

```python
# เปลี่ยนจาก (hardcoded):
VAULT_ROOT = r"D:\arjin-vault\Obs_Dr_Arjin"

# เป็น (flexible):
VAULT_ROOT = os.environ.get("AAOS_VAULT_ROOT", r"D:\arjin-vault\Obs_Dr_Arjin")
AAOS_ROOT  = os.environ.get("AAOS_ROOT", r"D:\arjin-vault\06-OUTPUT\AAOS")
```

---

## ขั้นตอนที่ 4 — Git Init & Push

```powershell
cd D:\AAOS-Academic-AI-OS

git init
git add .
git commit -m "🚀 Initial release: AAOS v1.0 — L1 to L6 complete"

# เชื่อม GitHub repo
git remote add origin https://github.com/arjin-nu/AAOS-Academic-AI-OS.git
git branch -M main
git push -u origin main
```

---

## ขั้นตอนที่ 5 — ตั้งค่า GitHub Repository

1. **Topics/Tags** (เพิ่มใน repo settings):
   ```
   multi-agent, agentic-ai, academic, langgraph, anthropic,
   claude, education, engineering, self-improving, python
   ```

2. **GitHub Pages** (optional): เปิด docs/ เป็น website

3. **Release**: สร้าง v1.0.0 release พร้อม changelog

4. **Zenodo DOI** (สำหรับ citation):
   - เปิด https://zenodo.org
   - Connect GitHub
   - Publish release → รับ DOI อัตโนมัติ

---

## 🎯 หลัง Push เสร็จ

URL ที่ได้: `https://github.com/arjin-nu/AAOS-Academic-AI-OS`

ใส่ใน Paper ได้เลย:
```
The source code is publicly available at:
https://github.com/arjin-nu/AAOS-Academic-AI-OS
DOI: 10.5281/zenodo.XXXXXXX
```
