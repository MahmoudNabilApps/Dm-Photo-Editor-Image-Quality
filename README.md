---
title: Real-ESRGAN API
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Real-ESRGAN Image Upscaling API

API لرفع جودة الصور باستخدام Real-ESRGAN يعمل 24/7 مجاناً على Hugging Face Spaces.

## Endpoints

- `GET /` - الصفحة الرئيسية
- `GET /health` - فحص حالة API
- `POST /upscale` - رفع جودة الصورة

## استخدام API

POST /upscale
Content-Type: application/json

{
"image": "base64_encoded_image_here"
}

## Features

- ✅ يعمل 24/7 بدون توقف
- ✅ مجاني تماماً
- ✅ رابط ثابت
- ✅ محسّن للـ CPU
