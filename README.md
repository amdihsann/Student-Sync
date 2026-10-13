# Student-Sync: AI Productivity Agent 🤖

Student-Sync is a Telegram-based AI assistant designed specifically to boost the productivity of IT students. This bot understands natural language to automatically log financial expenses and manage schedules.

This project was built as a Capstone Project for the **IBM SkillsBuild University Education** program (AI Agent for Programming Bootcamp). The development process utilized the *Full Vibe Coding* method with the assistance of an AI Agent (IBM Bob - Trial Version) within a short, intensive timeframe.

## 🎯 Core Features

1. **Automated Finance Tracker:**
   Extracts information such as amount, category, and description from text messages, and automatically logs the transaction into Google Sheets.

2. **Schedule Manager (Calendar):**
   Detects scheduling intents, converts times into the ISO 8601 standard format (Asia/Jakarta), and adds events directly to the user's Google Calendar.

3. **Multi-Action Parsing:**
   Capable of parsing and executing multiple distinct commands from a single sentence (e.g., *"I just bought gas for 20k, oh and by the way, I have a class tomorrow at 10 AM"*).

## 🛠️ Tech Stack & Architecture

* **Platform/Interface:** Telegram Bot API (Webhook)
* **Backend Framework:** FastAPI (Python) & Uvicorn
* **Tunneling:** Ngrok (to connect the local server to Telegram)
* **AI/LLM Provider:** Groq API (using the `qwen/qwen3.8-27b` model via Cloud Inference)
* **Third-Party Integrations:** Google Sheets API (`gspread`), Google Calendar API and Google Drive API

## 📝 Development Notes

This project focuses on establishing robust backend functionality (intent routing, AI reasoning, and external API connections) within the bootcamp's time constraints. Although currently a prototype, the core architecture is modularly designed using a Client-Server approach. This ensures that the heavy lifting of AI inference is handled via the cloud, preventing any computing burden on the local machine.

*Created for IBM SkillsBuild University Education.*